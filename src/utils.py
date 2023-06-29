"""Module containing helper functions to analyse networks generated with Bianconi-Barabasi model."""

import numpy as np
import matplotlib.pyplot as plt

from . import model

def get_node_degrees(net):
    """
    Get the degree over time for every node.

    Parameters
    ----------
    net : .network
        object of the class network
    
    Returns
    -------
    times : list
        list with the timesteps at which the node existed for every node
    node_degrees : list
        list with the degree over time for every node
    """
    # Check if the variable is a .network object
    if str(type(net))[-10:-2] != '.network':
        raise TypeError('The input variable is not an object of the class network')
    
    node_degrees = []
    times = []
    for i in net.graph.keys():
        k, t = net.get_degree_wrt_time(i)
        node_degrees.append(k)
        times.append(t)

    return times, node_degrees

def simulate(size, n_iterations, fitness_val):
    """
    Runs the model 'n_iterations' times.

    Parameters
    ----------
    size : int
        size of the generated network
    n_iterations : int
        number of times the network should be generated
    fitness_val : list
        list with the fitness values that will be used to generate the model
    
    Returns
    -------
    times : list
        list with the timesteps at which the node existed for every node for every run
    node_degrees : list
        list with the degree over time for every node for every run
    """
    # Check the type of n_iterations
    if type(n_iterations) != int:
        raise TypeError('The parameter n_bins should be an integer instead of %s.'%type(n_iterations).__name__)
    node_degrees = []
    times = []

    for _ in range(n_iterations):
        network = model.network()
        network.generator.fitness_data = fitness_val
        network.set_fitness_distr('data')
        network.generate_network(size)
        
        t, k = get_node_degrees(network)
        node_degrees.append(k)
        times.append(t)

    return times, node_degrees

def clone_shape(arr):
    """
    Creates a copy of a multidimensional list filled with zeros.
    """
    res = []
    for x in arr:
        res.append(np.zeros_like(x))
    return res

def get_means(arr):
    """
    Calculate average degrees for all nodes over time for multiple model runs.

    Parameters
    ----------
    arr : list
        list with the degree over time for every node for every run
    
    Returns
    -------
    means : list
        Average degree for all nodes over multiple runs
    """
    summ = clone_shape(arr[0])
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                summ[j][k] += arr[i][j][k]

    means = [[value / len(arr) for value in subarr] for subarr in summ]

    return means
