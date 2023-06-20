"""Module that contains the code for the Bianconi-Barabasi model."""

# Imports
import numpy as np

class network:
    """
    Framework for a network generated according to the Bianconi-Barabasi model.

    Attributes
    ----------
    graph : dict
        Representation of the network.
        Key correspond to the node ID
        Value is (list containing ID of the neighbouring nodes, fitness of the node)  
    m0 : int
        Number of nodes in the network at time zero
    m : int
        Number of nodes each new node links to
    tot_fitness : float
        Sum of the fitness multiplied with the degree for every node
    fitness_distr : str
        Distribution from which the fitness values are generated
    
    Methods
    -------
    set_up()
        Generate an initial network with m0 nodes.
    generate_fitness_value()
        Generate a fitness value from a given distribution.
    add_node()
        Add a new node to the network.
    generate_network(n)
        Generate a network with n nodes.
    """
    def __init__(self, m0=5, m=2):
        """
        Initialize an object of the class network.

        Parameters
        ----------
        m0 : int
            Number of nodes in the network at time zero, default 5
        m : int
            Number of nodes each new node links to, default 2
        """
        # Check the type of m0 and m
        if type(m0) != int:
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m0))
        if type(m) != int:
            raise TypeError('The parameter m should be an integer instead of %s.'%type(m))
        
        # Fitness
        self.tot_fitness = 0
        self.fitness_distr = 'uniform'

        # Graph
        self.m0 = m0
        self.m = m
        self.graph = {}
        # Create initial graph
        self.set_up(m0)

    def set_up(self, m0):
        """
        Generate an initial network with m0 nodes.

        Parameters
        ----------
        m0 : int
            Number of nodes at time zero
        """
        self.tot_fitness = 0
        self.graph = {}
        for i in range(m0):
            # Determine the fitness value of the node
            fitness = self.generate_fitness_value()
            self.graph[i] = [], fitness
        
        # Give every node 1 neighbour
        for i in range(m0):
            # Select a node to link with
            neighbor = np.random.randint(0, m0)
            if neighbor == i:
                neighbor = (neighbor + 1) % m0 
            self.graph[i][0].append(neighbor)
            self.graph[neighbor][0].append(i)

            # Update the total fitness of the network
            self.tot_fitness += self.graph[i][1]
            self.tot_fitness += self.graph[neighbor][1]
    
    def set_m0(self, m0):
        """
        Set the value for the number of initial nodes, m0.

        Parameters
        ----------
        m0 : int
            Number of nodes at time zero
        """
        # Check the type of m0
        if type(m0) != int:
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m0))
        self.m0 = m0
        # Regenerate the initial network with new value for m0
        self.set_up(m0)
    
    def set_m(self, m):
        """
        Set the value for the number of nodes a new node links to, m.

        Parameters
        ----------
        m : int
            Number of nodes each new node links to
        """
        # Check the type of m
        if type(m) != int:
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m))
        self.m = m
    
    def set_fitness_distr(self, distr):
        """
        Set the type of distribution from which fitness values are generated.

        Parameters
        ----------
        distr : str
            Fitness distribution
        """
        if distr != 'delta' and distr != 'uniform' and distr != 'exponential':
            raise NameError('Unkown distribution, %s, used. Options are "delta", "uniform" or "exponential". '%distr)
        self.fitness_distr = distr

    def generate_fitness_value(self):
        """
        Generate a fitness value from a given distribution.
        """
        # Delta distribution
        if self.fitness_distr == 'delta':
            return 1
        # Uniform distribution
        elif self.fitness_distr == 'uniform':
            return np.random.uniform()
        # Exponential distribution
        elif self.fitness_distr == 'exponential':
            return np.random.exponential(1)
        else:
            raise NameError('Unkown distribution, %s, used. Options are "delta", "uniform" or "exponential". '%self.fitness_distr)

    def add_node(self):
        """
        Add a new node to the network.
        """
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

    def generate_network(self, n):
        """
        Generate a network with n nodes.

        Parameters
        ----------
        n : int
            Total number of nodes in the network
        """
        # Check the type of n
        if type(n) != int:
            raise TypeError('The parameter n should be an integer instead of %s.'%type(n))
        # n should be larger than m0
        if n < self.m0:
            raise ValueError('Total number of nodes (%s) cannot be smaller than m0 (%s)'%(n, self.m0))
        
        while len(self.graph) < n:
            self.add_node()
            