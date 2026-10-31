"""Independent causal-attention arithmetic anchors for budget training."""
from copy import deepcopy

import pytest

from supernode_sim.config import Config
from supernode_sim.inference import build_inference
from supernode_sim.topology import Topology
from supernode_sim.training import build_training


def configuration(sequence, *, hidden=16, heads=4, head_dim=4, tp=1, ep=1,
                  checkpoint=False):
    return Config.model_validate({
        "model": {"name": "synthetic-attention-work", "total_params": 256,
                  "layers": 1, "hidden_dim": hidden, "num_heads": heads,
                  "head_dim": head_dim, "kv_heads": 1, "num_experts": ep},
        "hardware": {"npu_count": tp * ep, "su_size": tp * ep},
        "workload": {"scenario": "training", "batch_size": 2,
                     "sequence_length": sequence, "input_length": sequence,
                     "output_length": 1},
        "execution": {"tp": tp, "ep": ep,
                      "features": {"activation_checkpointing": checkpoint}},
    }).model_dump()


def graph(config, build=build_training):
    return build(config, Topology(config["hardware"], config["execution"]["precision"]))


@pytest.mark.parametrize("sequence,pairs_per_sequence", [(1, 1), (4, 10)])
@pytest.mark.parametrize("hidden,heads,head_dim", [(16, 4, 4), (32, 4, 4), (16, 2, 6)])
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_training_attention_triangle_heads_and_parallel_work(
        sequence, pairs_per_sequence, hidden, heads, head_dim, tp, ep):
    config = configuration(sequence, hidden=hidden, heads=heads, head_dim=head_dim,
                           tp=tp, ep=ep, checkpoint=True)
    events = graph(config).events
    # QK and AV each perform head_dim multiply-adds for every valid pair.
    attention_work = 2 * pairs_per_sequence * heads * head_dim * 4
    parameter_work = 2 * 256 * 2 * sequence
    expected_forward = attention_work + parameter_work
    for phase, multiplier in (("forward", 1), ("backward", 2), ("recompute", 1)):
        kernels = [event for event in events
                   if event.metadata.get("kernel") == "training_" + phase]
        assert len(kernels) == tp * ep
        assert sum(event.metadata["flops"] for event in kernels) == pytest.approx(
            multiplier * expected_forward)
        assert all(event.metadata["flops"] == pytest.approx(
            multiplier * expected_forward / (tp * ep)) for event in kernels)
        for event in kernels:
            assert event.work[event.metadata["rank"] + ".compute"] == event.metadata["flops"]


@pytest.mark.parametrize("sequence", [1, 7])
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 2)])
def test_budget_training_forward_matches_budget_prefill_attention(sequence, tp, ep):
    config = configuration(sequence, hidden=32, heads=4, head_dim=6, tp=tp, ep=ep)
    training = graph(config)
    inference_config = deepcopy(config)
    inference_config["workload"]["scenario"] = "inference"
    inference = graph(inference_config, build_inference)
    forward_work = sum(event.metadata["flops"] for event in training.events
                       if event.metadata.get("kernel") == "training_forward")
    prefill_work = sum(event.metadata["flops"] for event in inference.events
                       if event.metadata.get("kernel") == "attention_dense")
    assert forward_work == pytest.approx(prefill_work)
