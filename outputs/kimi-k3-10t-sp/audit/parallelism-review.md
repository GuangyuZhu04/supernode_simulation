# Megatron SP16 / SP32 independent review

The selected configurations are **TP16 + SP16 + EP64** and **TP32 + SP32 + EP32**. SP shares the TP communicator; it does not multiply the 1024-card world size. Weights retain tensor sharding, and expert count stays 3328. Each TP/SP group spans respectively 2 or 4 eight-card nodes, entirely inside one 64-card scale-up rack. A fixed TP coordinate's EP communicator visits 64 or 32 source lanes, with 4 or 2 lanes per rack and 16 racks in both cases.

## Attention and cache semantics

Each attention and FFN block gathers sequence-sharded hidden rows before its tensor-parallel projection and reduce-scatters the hidden result afterward. PreNorm, PostNorm and AttnRes residual/workspace rows can remain sequence-sharded. Attention itself still receives each source sequence's complete ordered valid token rows. KDA has 96/TP local heads and uses its existing full-sequence recurrent/chunk kernel; no cross-SP KDA prefix scan is required. MLA has 96/TP query heads; its compressed 576-element KV cache remains replicated across TP ranks. SP does **not** divide compressed MLA KV capacity by 16 or 32. The expanded-cache alternative remains head-sharded by TP.

This differs from independent context/sequence parallelism, which would partition attention context and require an explicit MLA ring/gather or distributed-softmax reduction and a KDA recurrent-state/prefix-scan algorithm. Such semantics are not selected here. For reference, independent TP1/SP16/EP64 and TP1/SP32/EP32 would already need 231.381 GB and 394.335 GB static HBM per card, exceeding 144 GB before caches and activations.

## Boundary and short-sequence handling

Vocabulary-sharded embedding produces a partial hidden vector on each TP rank, so the selected SP model includes its reduce-scatter. Before the output vocabulary projection it gathers only selected last-position hidden rows, preserving the prior experiment's prefill last-position-logits optimization. Full logits gathering, token sampling and queue latency remain outside the existing model.

Equal-size SP collectives pad each nonempty EP source lane's row count to a TP multiple. Valid attention/FFN work and persistent caches exclude padding. Long prefill lengths divide both TP factors exactly; decode owns one valid query row per nonempty EP lane because batch <= 32 and EP >= 32. A decode buffer therefore contains TP padded rows for one valid query. This avoids fractional row ownership and materially increases the padded decode all-gather/reduce-scatter payload. A specialized ragged or SP-disabled decode implementation would yield different results and requires its own declared policy.

## Algebraic checks

Let `N` denote total valid source rows, `P=sum_e ceil(N_e/TP)*TP` padded source rows, and `Q=sum_e ceil(active_sequences_e/TP)*TP` selected-output rows after padding. Then:

- Main-body all-gather and reduce-scatter each carry `2 * 93 * (TP-1) * P * 7168 * 2` bytes across the entire system.
- Embedding reduce-scatter carries `(TP-1) * P * 7168 * 2`; selected-output all-gather carries `(TP-1) * Q * 7168 * 2`.
- The two main-body AG+RS pairs replace two AR operations per layer. With identical valid, unpadded shapes their combined byte count equals two AR operations at the same TP factor.
- Existing balanced MoE routing repeats a full 3584-element latent vector independently for each TP slice. Raw dispatch+combine assignments are `2 * 92 * TP * N * top_k * 3584 * 2` bytes; local assignments use no network. Network fraction is `(EP-1)/EP`, same-rack fraction `(64/TP-1)/EP`, and cross-rack fraction `(EP-64/TP)/EP`.
- All TP/SP payload remains on scale-up. Cross-rack MoE payload uses the existing four-hop SO topology. Doubling TP at fixed 1024 cards doubles raw TP-slice MoE replication and cross-rack MoE payload under this inherited dispatch model.

| Case | Static HBM/card | One-step decode payload/query | Cross-rack payload/query |
|---|---:|---:|---:|
| TP8, EP128, top32 baseline | 32.596955412 GB | 0.372334592 GB | 0.316538880 GB |
| TP8, EP128, top64 baseline | 32.596955412 GB | 0.707338240 GB | 0.633077760 GB |
| TP16/SP16, EP64, top32 | 29.310190110 GB | 1.951531008 GB | 0.633077760 GB |
| TP16/SP16, EP64, top64 | 29.310190110 GB | 2.616262656 GB | 1.266155520 GB |
| TP32/SP32, EP32, top32 | 27.666807459 GB | 6.627131392 GB | 1.266155520 GB |
| TP32/SP32, EP32, top64 | 27.666807459 GB | 7.935492096 GB | 2.532311040 GB |

Static includes FP8 weights, per-tensor 128-element/4-byte scales, 8 GB runtime and 256 MB system workspace; it excludes runtime caches and activations. Baseline omits embedding/output boundary communication in the existing engine. Selected SP adds the required boundaries, so the algebra records these separately. Payloads are sums across all cards, not per-card traffic. Multiply one-query decode payload by batch size; context length changes local KV/HBM work but does not change resident-cache decode network bytes under this parallel layout.

`derive_traffic.py` reproduces `traffic-formulas.json`: all 96 configurations with separate prefill/decode planned demands, AG, RS, dispatch/combine, SU/SO and padding counts. `static-capacity-comparison.json` records exact `ParameterTensor.local_elements` accounting. `traffic-formulas-candidates.json` is the earlier unpadded semantic comparison and must not be used as the selected experiment's expected decode traffic.
