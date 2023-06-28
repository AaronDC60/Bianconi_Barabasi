"""File that contains the code for the Bianconi-Barabasi model."""

# Imports
import os
import shutil
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import imageio

import fitness2 as fitness

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
    generator : src.fitness.generator
        Object that can generate fitness values from different generators
    
    Methods
    -------
    set_up()
        Generate an initial network with m0 nodes.
    set_m0(m0)
        Set the value for the number of initial nodes, m0.
    set_m(m)
        Set the value for the number of nodes a new node links to, m.
    set_fitness_distr(distr)
        Set the type of distribution from which fitness values are generated.
    generate_fitness_value()
        Generate a fitness value from a given distribution.
    add_node()
        Add a new node to the network.
    generate_network(n)
        Generate a network with n nodes.
    get_largest_node()
        Get the node ID of the node with the highest degree.
    get_degree_wrt_time(node)
        Get the degree of a node at every point in time.
    get_degree_distr(n_bins)
        Get the degree distribution of the network.
    convert_to_networkx(t)
        Make a representation of the graph at time t in networkx.
    create_gif(filename, time_steps)
        Create a GIF animation of the network evolution over time.
    """
    def __init__(self, m0=3, m=2):
        """
        Initialize an object of the class network.

        Parameters
        ----------
        m0 : int, default 3
            Number of nodes in the network at time zero
        m : int, default 2
            Number of nodes each new node links to
        """
        # Check the type of m0 and m
        if type(m0) != int:
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m0).__name__)
        if type(m) != int:
            raise TypeError('The parameter m should be an integer instead of %s.'%type(m).__name__)
        
        # Fitness
        self.fitness_values = []
        self.tot_fitness = 0
        self.generator = fitness.generator()

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
        self.fitness_values = []
        self.tot_fitness = 0
        self.graph = {}
        for i in range(m0):
            # Determine the fitness value of the node
            fitness = self.generator.generate_value()
            self.fitness_values.append(fitness)
            self.graph[i] = [], fitness
        
        # Give every node 1 neighbour
        for i in range(m0):
            # Select a node to link with
            neighbor = np.random.randint(0, m0)
            if neighbor == i:
                # No-self links
                neighbor = (neighbor + 1) % m0
            if neighbor not in self.graph[i][0]:
                # Only creat a link if it doesn't already exist
                self.graph[i][0].append(neighbor)
                self.graph[neighbor][0].append(i)

                # Update the total fitness of the network
                self.tot_fitness += self.graph[i][1]
                self.tot_fitness += self.graph[neighbor][1]


    def set_up_from_data(self, m0):
        """
        Generate an initial network with m0 nodes with fixed fitness distribution from data.

        Parameters
        ----------
        m0 : int
            Number of nodes at time zero
        """
        self.graph = {}

        for i in range(m0):
            # Determine the fitness value of the node
            fitness = self.generator.fitness_data[i]
            self.graph[i] = [], fitness

        # Give every node 1 neighbour
        for i in range(m0):
            # Select a node to link with
            neighbor = np.random.randint(0, m0)
            if neighbor == i:
                # No-self links
                neighbor = (neighbor + 1) % m0
            if neighbor not in self.graph[i][0]:
                # Only creat a link if it doesn't already exist
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
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m0).__name__)
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
            raise TypeError('The parameter m0 should be an integer instead of %s.'%type(m).__name__)
        self.m = m
    
    def set_fitness_distr(self, distr):
        """
        Set the type of distribution from which fitness values are generated.

        Parameters
        ----------
        distr : str
            Fitness distribution
        """
        self.generator.set_current_distribution(distr)
        # Regenerate the initial network with new fitness distribution
        self.set_up(self.m0)

    def add_node(self):
        """
        Add a new node to the network.
        """
        # Determine the fitness value of the new node
        node_id = len(self.graph)
        fitness = self.generator.generate_value()
        self.fitness_values.append(fitness)
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

    def add_node_from_data(self):
        """
        Add a new node to the network from a data with fitness distribution.
        """
        # Determine the fitness value of the new node
        node_id = len(self.graph)
        fitness = self.generator.fitness_data[node_id]

        #print(self.graph)

        #print(np.sum(self.tot_fitness))
        # Construct the pdf of the network
        pdf = [(self.graph[i][1] * len(self.graph[i][0]))/self.tot_fitness for i in self.graph.keys()]

        #print(pdf)

        # Determine the nodes to which the new node will link
        neighbors = np.random.choice(list(self.graph.keys()), self.m, replace=False, p=pdf)
        
        # Add new node to the network
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
        
        Returns
        -------
        graph : dict
            The network
        """
        # Check the type of n
        if type(n) != int:
            raise TypeError('The parameter n should be an integer instead of %s.'%type(n).__name__)
        # n should be larger than m0
        if n < self.m0:
            raise ValueError('Total number of nodes (%s) cannot be smaller than m0 (%s)'%(n, self.m0))
        
        while len(self.graph) < n:
            self.add_node()
        return self.graph
    
    def generate_network_from_data(self, n):
        """
        Generate a network with n nodes .

        Parameters
        ----------
        n : int
            Total number of nodes in the network
        
        Returns
        -------
        graph : dict
            The network
        """
        # Check the type of n
        if type(n) != int:
            raise TypeError('The parameter n should be an integer instead of %s.'%type(n))
        # n should be larger than m0
        if n < self.m0:
            raise ValueError('Total number of nodes (%s) cannot be smaller than m0 (%s)'%(n, self.m0))

        while len(self.graph) < n:
            self.add_node_from_data()

        return self.graph

    def get_largest_node(self):
        """
        Get the node ID of the node with the highest degree.

        Returns
        -------
        node_id : int
            ID of the node with the highest degree
        """
        node_id = 0
        # Run over all the node in the network and check which one has the most neighbours
        for node in self.graph.keys():
            if len(self.graph[node][0]) > len(self.graph[node_id][0]):
                node_id = node
        return node_id

    def get_degree_wrt_time(self, node):
        """
        Get the degree of a node at every point in time.

        Parameters
        ----------
        node : int
            ID of the node for which the change degree should be returned
        
        Returns
        -------
        k_t : numpy.ndarray
            Array with the degree of the node at every point in time
        t : numpy.ndarray
            Array with timepoint that match k_t
        """
        # Check if the node variable is an integer
        if type(node) != int:
            raise TypeError('The parameter node should be an integer, not %s.'%type(node).__name__)
        # Check if the node id exists
        if node >= len(self.graph):
            raise ValueError('Given node ID does not exist.')
        
        # Create array with all the timesteps from at which the node exists
        # First m0 nodes are created at t=0
        # Last timestep is the total number of nodes minus m0 because 1 node added per timestep
        t = np.arange(max(0, node - self.m0 + 1), len(self.graph) - self.m0 + 1)
        k_t = np.zeros(len(t))

        # Loop over all neighbours to determine when the degree changed
        for neighbor in self.graph[node][0]:
            # Time at which the neighbour was created
            time = max(0, neighbor - self.m0 + 1)
            if time < t[0]:
                # Neighbour was created earlier, this is one of the m links created at the first timestep
                # Increase degree at every timestep
                k_t += 1
            else:
                # Increase degee from when the neighbour linked to this node
                k_t[time-t[0]:] += 1
        return k_t, t

    def get_degree_distr(self, n_bins = 20):
        """
        Get the degree distribution of the network.

        Parameters
        ----------
        n_bins : int
            Number of bins 

        Returns
        -------
        p_k : numpy.ndarray
            Probability distribution (y-axis)
        bin_centers : numpy.ndarray
            Degrees (x-axis)
        """
        # Check the type of n_bins
        if type(n_bins) != int:
            raise TypeError('The parameter n_bins should be an integer instead of %s.'%type(n_bins).__name__)
        
        # Determine the degree of every node (length of the list with neighbours) in the network
        degree = np.empty(len(self.graph))
        for node in self.graph.keys():
            degree[node] = len(self.graph[node][0])
        
        # Create the (log)bins
        log_bins = np.logspace(np.log10(min(degree)), np.log10(max(degree)), num = n_bins)
        p_k, bin_edges = np.histogram(degree, bins=log_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return p_k, bin_centers

    def convert_to_networkx(self, t):
        """
        Make a representation of the graph at time t in networkx.

        Parameters
        ----------
        t : The time at which the graph should be returned

        Returns
        -------
        G : networkx.classes.graph.Graph
            Graph at time t
        """
        # Check if t is an integer (discrete timesteps)
        if type(t) != int:
            raise TypeError('The parameter t should be an integer instead of %s.'%type(t).__name__)
        # Check if t does not exceed the size of the network
        if t + self.m0 > len(self.graph):
            raise ValueError('Timestep t exceeds the size of the system. There are only %d timesteps.'%(len(self.graph - self.m0)))
        
        # Create Graph object
        G = nx.Graph()

        # Loop over all nodes that were created at or before time t
        for i in range(self.m0 + t):
            # Add node
            G.add_node(i, fitness=self.graph[i][1])
            # Add link with neighbour if neighbour was present at time t
            for neighbor in self.graph[i][0]:
                if neighbor < self.m0 + t:
                    G.add_edge(i, neighbor)
        return G
    
    def create_gif(self, filename, time_steps=200):
        """
        Create a GIF animation of the network evolution over time.

        Parameters
        ----------
        filename : str
            Name of the GIF file as which the animation should be saved
        time_steps : int, default 200
            Number of timesteps for which the network evolution should be made
        """
        # Check if the time_steps parameter is an integer
        if type(time_steps) != int:
            raise TypeError('The parameter time_steps should be an integer instead of %s.'%type(time_steps).__name__)
        # Check if t does not exceed the size of the network
        if time_steps + self.m0 > len(self.graph):
            raise ValueError('Timesteps exceeds the size of the system. There are only %d timesteps.'%(len(self.graph) - self.m0))
        # Check if it is a valid filename
        if type(filename) != str:
            raise TypeError('The parameter filename should be a string instead of %s.'%type(filename).__name__)
        if filename[-4:] != '.gif':
            raise TypeError('The filename should end in .gif')

        # Make temporary folder to store the images
        os.mkdir('tmp')
        # Visualize the graph at every timestep
        G = self.convert_to_networkx(0)
        pos = nx.spring_layout(G)

        # Plot graph at t=0
        fig = plt.figure(figsize=(8, 6))
        nx.draw(G, pos=pos, with_labels=True, node_color=self.fitness_values[:self.m0], cmap='coolwarm')
        plt.savefig('tmp/image_' + str(self.m0 -1) + '.png')
        plt.close(fig)


        for i in range(self.m0, time_steps + self.m0):
            # Add node
            G.add_node(i, fitness=self.graph[i][1])

            # Add link with neighbour if neighbour was present at time t
            for neighbor in self.graph[i][0]:
                if neighbor < i:
                    G.add_edge(i, neighbor)
            pos = nx.spring_layout(G)

            # Calculate node degrees
            node_degrees = dict(G.degree())

            # Extract node sizes based on degrees
            node_sizes = [50 * node_degrees[node] for node in G.nodes()]

            # Draw the graph at time=i
            fig = plt.figure(figsize=(8, 6))
            nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=self.fitness_values[:i+1],\
                     cmap='coolwarm', vmin=min(self.fitness_values), vmax=max(self.fitness_values), edge_color='gray', alpha=0.7)
            # Store image in temporary directory
            plt.savefig('tmp/image_' + str(i) + '.png')
            plt.close(fig)
        
        # Make gif
        frames = []
        for i in range(self.m0 - 1, time_steps + self.m0):
            image = imageio.v2.imread(f'tmp/image_'+ str(i) + '.png')
            frames.append(image)

        # Save gif
        imageio.mimsave(filename, frames, duration = 500)
        # Delete temporary directory
        shutil.rmtree('tmp')
