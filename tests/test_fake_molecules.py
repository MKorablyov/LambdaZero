from tests.fake_molecules import get_list_edge_indices_for_a_ring


def test_get_list_edge_indices_for_a_ring():
    number_of_nodes = 3
    expected_list = [[2, 0], [0, 2], [0, 1], [1, 0], [1, 2], [2, 1]]
    computed_list = get_list_edge_indices_for_a_ring(number_of_nodes)

    assert expected_list == computed_list