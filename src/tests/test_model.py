"""Test file for the network generated according to the Bianconi-Barabasi model."""

# Imports
import pytest
from src import model

def check_connections(network):
    """
    Check if all the links in the network are stored correctly.
    """
    for node in network.graph.keys():
        # Every node needs to have at least 1 neighbour
        assert(len(network.graph[node][0]) != 0)
        for neighbour in network.graph[node][0]:
            # Node should be present in the neighbouring nodes list of its own neighbours
            assert(node in network.graph[neighbour][0])

def check_fitness(network):
    """
    Check if the total fitness in the network is correctly stored
    """
    total_fitness = 0
    for node in network.graph.keys():
        total_fitness += (network.graph[node][1] * len(network.graph[node][0]))
    assert(abs(total_fitness - network.tot_fitness) < 1E-10)

def test_initialization():
    """
    Check if the initialization of the network is done correctly.
    """
    # Incorrect type for m0 and m
    with pytest.raises(TypeError):
        model.network(m0=1.5)
        model.network(m=1.5)

    network = model.network()
    assert(len(network.graph) == network.m0)

def test_set_up():
    """
    Check if the set up of the network is done correctly
    """
    network = model.network()
    m0 = 6
    m = 3
    distr = 'delta'

    # Expected errors
    with pytest.raises(TypeError):
        network.set_m0('wrong type')
        network.set_m(5.4)
    with pytest.raises(NameError):
        network.set_fitness_distr('incorrect distribution')

    # Check if the set functions work
    network.set_m0(m0)
    network.set_m(m)
    network.set_fitness_distr(distr)

    assert(len(network.graph) == m0)
    assert(network.m == m)
    assert(network.generator.current_distribution == distr)

    check_connections(network)
    check_fitness(network)

def test_add_node():
    """
    Check if adding a node updates the network correctly.
    """
    network = model.network()
    length = len(network.graph)
    # Check the length of the network
    assert(length == network.m0)

    network.add_node()

    # Length should be increased by one
    assert(len(network.graph) == length + 1)

    check_connections(network)
    check_fitness(network)

def test_generate_network():
    """
    Check if generating a network of n nodes works.
    """
    n = 100
    network = model.network()
    # Expected errors
    with pytest.raises(TypeError):
        network.generate_network(5.5)
    with pytest.raises(ValueError):
        network.generate_network(network.m0 - 1)
    network.generate_network(n)

    # Check the length of the network
    assert(len(network.graph) == n)
    
    check_connections(network)
    check_fitness(network)

def test_finding_largest_node():
    """
    Check if the function indeed returns the node with the highest degree.
    """
    # Generate a network
    network = model.network()
    network.set_fitness_distr('uniform')
    network.generate_network(1000)

    largest_node = network.get_largest_node()

    for node in network.graph.keys():
        if node != largest_node:
            # degree should be lower or equal
            assert(len(network.graph[node][0]) <= len(network.graph[largest_node][0]))
        else:
            # degree should be the same
            assert(len(network.graph[node][0]) == len(network.graph[largest_node][0]))

def test_degree_wrt_time():
    """
    Check the degree as a function of time.
    """
    n = 1000
    # Generate a network
    network = model.network()
    network.set_fitness_distr('uniform')
    network.generate_network(n)

    # Expected errors
    with pytest.raises(TypeError):
        network.get_degree_wrt_time(1.5)
    with pytest.raises(ValueError):
        network.get_degree_wrt_time(2000)
    
    # Check degree as a function of time for the m0 nodes
    for i in range(network.m0):
        k_t, t = network.get_degree_wrt_time(i)
        # First timestep should be 0
        assert(t[0] == 0)
        # Last timestep should be n-m0
        assert(t[-1] == n - network.m0)
        # Final degree should be the total number of neighbours
        assert(k_t[-1] == len(network.graph[i][0]))
    
    # Check another node that was created later
    id = 10
    k_t, t = network.get_degree_wrt_time(id)
    # First timestep should be id - m0
    assert(t[0] == id - network.m0 + 1)
    # Last timestep should be n-m0
    assert(t[-1] == n - network.m0)
    # First degree should be m
    assert(k_t[0] == network.m)
    # Final degree should be the total number of neighbours
    assert(k_t[-1] == len(network.graph[id][0]))
    