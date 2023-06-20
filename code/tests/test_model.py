"""Test file for the network generated according to the Bianconi-Barabasi model."""

# Imports
import pytest
import model

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
    assert(network.fitness_distr == distr)

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
