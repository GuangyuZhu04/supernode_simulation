# Isolated synchronous collective-round representation

Final runtime: code/runtime-synchronous-rounds/ (18 modules).
Manifest: code/runtime-synchronous-rounds-manifest.json.
Final runtime ID: sha256:78919546d7eb336972226e9f71cc396718eafeafc5999327c398eeb025c40620.
The original frozen runtime (sha256:447b4248c25385c9ca971fb1b070aa195ff2397136f5b75b48881bd615ee3645) remains unchanged in code/runtime/.

Only topology.py and streaming.py differ from the original numerical runtime.
The change is an exact event representation for this fixed experiment, not an analytic substitution for discrete-event simulation.

## Conditions and reasoning

1. The structured DP1/PP1 compiler places a full barrier around every communication stage; compute kernels do not overlap these stages. Independent communicators use disjoint NPUs.
2. Within a ring or pairwise round, each NPU has at most one transmit and one receive. All positive transfers have identical payload and the same one-way startup latency. The logical full-mesh all-SU model has independent directed links and uniform endpoint injection rates, with no shared host NIC.
3. Each matching transfer therefore starts and ends simultaneously. Endpoint HBM bandwidth is at least twice SU injection bandwidth, so simultaneous transmit/receive HBM traffic cannot break this symmetry.
4. One aggregate round event has the union of original slots, the same per-resource work, the sum of original staging bytes in each device memory pool, the same startup, and summed payload/physical-link bytes. This conserves normalized service, resource busy time, buffer peaks and barriers. Both representations continue through the same fluid discrete-event scheduler.
5. Atomic aggregate admission is safe only when all messages fit together. The exact normalized model, precision and memory-feature hashes are pinned, together with NPU=SU=1024, TP*EP=1024, TP8/16/32, top32/64, uniform contexts at most 1,048,576, batch at most min(32, EP), output length 2, runtime at most 8 GB, workspace at most 256 MB and HBM at least 288 GB. A conservative communication-stage resident bound is below 191 GB/device: weight/quantization/runtime, compressed KV/KDA, retained AttnRes residuals, SP gather and maximum send/receive staging. Larger transient compute/AttnRes workspaces retain original admission and failure behavior. The optimization falls back to the original collective for other configurations or nonuniform positive payloads.

All bounds use SI bytes. This proof is conditional on the logical nonblocking full-mesh hardware abstraction used in the public-spec experiment; it adds no claim about a physical 960/970 switch layout.

## Evidence

- all-cases-eligibility.json evaluates the actual frozen Python guard AST against all 288 resolved configurations, checks the topology/rate conditions, and records its source and SHA.
- A6 TP8/top32/128k/b1 native reference and prototype match all physical fields; modest runtime gain reflects its sparse single-sequence EP traffic.
- A7 SP32/top64/1M/b32 native reference and prototype match TTFT/TPOT, phase payload, physical-domain bytes, per-resource work/busy time, HBM peak/final/category and diagnostics at relative tolerance 1e-10, absolute 1e-7. Prototype wall time 94.20 s versus native 200.45 s (2.13x). The native 36,648,968 scheduler events become 2,329,736 events, with identical physical results.
- Final guarded-runtime A6 replay captures and verifies the immutable package SHA before numerical imports. The changes after prototype verification only restrict eligibility; the aggregate-round mathematics is unchanged.
- Prototype-v1 and intermediate guard revisions have separate audit copies and IDs. No native reference case or original runtime was overwritten.

Event counts intentionally differ because synchronized transfers have a compressed representation. Metadata records generated coalesced rounds and the corresponding original transfer count. These are generated graph counts only: normal component-macro replays conserve timing/traffic without generating the same graph again. Never interpret these counters as the full workload transport volume.

Reproduce from repository root:

    .\.venv\Scripts\python.exe outputs/ascend960-970-kimi-10t/code/verify_synchronous_eligibility.py
    .\.venv\Scripts\python.exe outputs/ascend960-970-kimi-10t/code/benchmark_synchronous_rounds.py a6-tp8-top32-128k-b1

The benchmark reads native pilot artifacts and writes only into this audit directory. Preserve native reference copies if the main cases directory is regenerated with the new runtime ID.
