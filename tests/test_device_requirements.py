from supernode_sim.analysis import device_requirements
from supernode_sim.core import Event, Graph


def test_mixed_precision_and_endpoint_footprints_are_per_device():
    graph = Graph(events=[
        Event('matrix', work={'npu0.compute': 200, 'npu0.hbm': 80}, metadata={'working_set_bytes': 20}),
        Event('scalar', work={'npu0.compute.fp32': 12}, metadata={'working_set_bytes': 30}),
        Event('other', work={'npu1.compute.calibrated.fp16': 90}, metadata={'working_set_bytes': 50}),
        Event('transfer', metadata={'src': 'npu1', 'dst': 'npu0', 'payload_bytes': 12}),
        Event('host', metadata={'src': 'cpu0', 'dst': 'npu1', 'payload_bytes': 40}),
    ])
    result = device_requirements(graph, 3, {'npu0.hbm': 80, 'npu1.hbm': 64})
    assert result['npu0'] == {'hbm_access_bytes': 80, 'max_operator_working_set_bytes': 30, 'largest_transfer_payload_bytes': 12}
    assert result['npu1'] == {'hbm_access_bytes': 64, 'max_operator_working_set_bytes': 50, 'largest_transfer_payload_bytes': 40}
    assert result['npu2'] == {'hbm_access_bytes': 0, 'max_operator_working_set_bytes': 0, 'largest_transfer_payload_bytes': 0}
