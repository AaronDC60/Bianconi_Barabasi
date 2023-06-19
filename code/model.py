"""Module that contains the code for the Bianconi-Barabasi model."""

# Imports
import numpy as np

class network:
    def __init__(self, m0):

        self.tot_fitness = 0
        self.fitness_distr = 'uniform'

        self.m = 2
        # Graph with m0 nodes, each with one link
        self.graph = {}
        for i in range(m0):
            # Determine the fitness value of the node
            fitness = self.generate_fitness_value()
            # Select a node to link with
            neighbor = np.random.randint(0, m0)
            if neighbor == i:
                neighbor = (neighbor + 1) % m0 

            self.graph[i] = [neighbor], fitness
            self.tot_fitness += fitness

    def generate_fitness_value(self):

        if self.fitness_distr == 'delta':
            return 1
        
        elif self.fitness_distr == 'uniform':
            return np.random.uniform()

    def add_node(self):

        # Determine the fitness value of the new node
        node_id = len(self.graph)
        fitness = self.generate_fitness_value()
        # Construct the pdf of the network
        pdf = [(self.graph[i][1] * len(self.graph[i][0]))/self.tot_fitness for i in self.graph.keys()]

        # Determine the nodes to which the new node will link
        neighbors = np.random.choice(list(self.graph.keys()), self.m, replace=False, p=pdf)
        
        # Add new node to the network + update total fitness
        self.tot_fitness += (self.m * fitness)
        self.graph[node_id] = list(neighbors), fitness
        for node in neighbors:
            self.graph[node][0].append(node_id)
            self.tot_fitness += self.graph[node][1]

    def generate_network(self, n, m):
        self.m = m
        while len(self.graph) < n:
            self.add_node()
