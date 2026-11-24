"""Run from the repository root with python -m examples.bandwidth_overlap."""
from supernode_sim.core import Event, Graph, Resource
from supernode_sim.engine import simulate
from supernode_sim.topology import Topology


def single_transfer(schedule=()):
    link = Resource("link", 10e9, "byte/s", capacity_schedule=schedule)
    return Graph(resources={"link": link}, events=[Event("send", work={"link": 100e6})])


def overlap(shared, compute_ms):
    links = [
        {"id": f"branch{i}", "src": f"npu{i}", "dst": f"npu{i+2}", "domain": "SU",
         "bandwidth_Bps": 10e9, "latency_s": 0,
         "shared_group": "egress" if shared else None}
        for i in range(2)
    ]
    topology = Topology({"topology": "explicit", "npu_count": 4, "su_size": 4,
                         "explicit_links": links}, {})
    transfers = [topology.transfer(link["id"], link["src"], link["dst"], 50e6) for link in links]
    compute = topology.resources["npu0.compute"]
    return Graph(resources=topology.resources, events=[
        *transfers, Event("compute", work={compute.id: compute.capacity * compute_ms / 1000}),
        Event("communication_done", deps=tuple(event.id for event in transfers)),
        Event("task_done", deps=("communication_done", "compute")),
    ])


if __name__ == "__main__":
    for label, schedule in (("fixed", ()), ("scheduled", ((.004, 5e9), (.008, 10e9)))):
        result = simulate(single_transfer(schedule))
        print(f"100 MB, {label:9s} {1000 * result['duration_s']:5.1f} ms")
    print("egress       compute_ms   communication_ms   task_ms")
    for compute_ms in (12, 3):
        for shared in (True, False):
            result = simulate(overlap(shared, compute_ms))
            ends = {row["id"]: row["end_s"] * 1000 for row in result["trace"]}
            print(f"{'shared' if shared else 'independent':12} {compute_ms:10} "
                  f"{ends['communication_done']:18.1f} {ends['task_done']:9.1f}")
