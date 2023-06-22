"""File that contains to generate (fitness)values from different distributions."""

# Imports
import numpy as np

class generator:
    """
    Attributes
    ----------
    current_distribution : str
        'active' distribution from which values are being sampled
    distributions : list
        list with all the possible distributions
    rate_exp : float
        parameter for the exponential distribution
    rate_poisson : float
        parameter for the poisson distribution
    a : float
        alpha parameter for the beta distribution
    b : float
        beta parameter for the beta distribution
    theta : float
        parameter for the distribution used to get BE condensation
    
    Methods
    -------
    set_param_exp(rate)
        Set the parameter (rate) for the exponential distribution.
    set_param_poisson(rate)
        Set the parameter for the poisson distribution.
    set_param_beta(a, b)
        Set the parameters for the beta distribution.
    set_param_be(theta)
        Set the parameter for the distribution used to get BE condensation.
    set_current_distribution(distr)
        Set the current distribution from which values are sampled.
    from_uniform()
        Sample value from uniform distribution.
    from_exponential()
        Sample value from exponential distribution.
    from_poisson()
        Sample value from poisson distribution.
    from_beta()
        Sample value from beta distribution.
    from_be()
        Sample value from distribution which is used to get BE condensation.
    generate_value()
        Sample value from current distribution.
    """
    def __init__(self):
        """
        Initialize a object that can generate fitness values from different distributions.
        """
        self.current_distribution = 'delta'
        self.distributions = ['delta', 'uniform', 'exponential', 'poisson', 'beta', 'be']
        # Parameter (rate) for exponential distribution
        self.rate_exp = 1
        # Parameter (expected number of events in a time-interval) for poisson distribution
        self.rate_poisson = 1
        # Parameters for beta distribution
        self.a = 1
        self.b = 1
        # Parameter for distribution used to get BE condensation
        self.theta = 1
    
    def set_param_exp(self, rate):
        """
        Set the parameter (rate) for the exponential distribution.

        Parameters
        ----------
        rate : float, (>0)
            new decay rate of the distribution
        """
        # Check if the variable is an integer or a float
        if type(rate) !=  int and type(rate) != float:
            raise TypeError('Invalid type for the variable rate, %s. Expected int or float.'%type(rate))
        # Check if the variable is positive
        if rate < 0:
            raise ValueError('The rate should be a non-negative value.')
        self.rate_exp = rate

    def set_param_poisson(self, rate):
        """
        Set the parameter for the poisson distribution.

        Parameters
        ----------
        rate : float, (>0)
            new parameter of the distribution    
        """
        # Check if the variable is an integer or a float
        if type(rate) !=  int and type(rate) != float:
            raise TypeError('Invalid type for the variable rate, %s. Expected int or float.'%type(rate))
        # Check if the variable is positive
        if rate < 0:
            raise ValueError('The variable should be a non-negative value.')
        self.rate_poisson = rate
    
    def set_param_beta(self, a, b):
        """
        Set the parameters for the beta distribution.

        Parameters
        ----------
        a : float
            alpha parameter
        b : float
            beta parameter
        """
        # Check if the variables are integers or floats
        if type(a) !=  int and type(a) != float:
            raise TypeError('Invalid type for the variable a, %s. Expected int or float.'%type(a))
        if type(b) !=  int and type(b) != float:
            raise TypeError('Invalid type for the variable b, %s. Expected int or float.'%type(b))
        # Check if the variables are positive
        if a < 0:
            raise ValueError('The variable, a, should be a non-negative value.')
        if b < 0:
            raise ValueError('The variable, b, should be a non-negative value.')
        self.a = a
        self.b = b

    def set_param_be(self, theta):
        """
        Set the parameter for the distribution used to get BE condensation.

        Parameters
        ----------
        theta : float, (>0)
            new parameter of the distribution  
        """
        # Check if the variable is an integer or a float
        if type(theta) !=  int and type(theta) != float:
            raise TypeError('Invalid type for the variable theta, %s. Expected int or float.'%type(theta))
        # Check if the variable is positive
        if theta < 0:
            raise ValueError('The variable should be a non-negative value.')
        self.theta = theta

    def set_current_distribution(self, distr):
        """
        Set the current distribution from which values are sampled.

        Parameters
        ----------
        distr : str
            distribution from which values should be sampled
        """
        if distr not in self.distributions:
            raise NameError('Unkown distribution, %s, used. Options are %s.'%(distr, self.distributions))
        self.current_distribution = distr
    
    def from_uniform(self):
        """
        Sample value from uniform distribution.

        Returns
        -------
        x : float
            Value sampled from a uniform distribution
        """
        return np.random.uniform()

    def from_exponential(self):
        """
        Sample value from exponential distribution.

        Returns
        -------
        x : float
            Value sampled from an exponential distribution
        """
        return np.random.exponential(self.rate_exp)

    def from_poisson(self):
        """
        Sample value from poisson distribution.

        Returns
        -------
        x : float
            Value sampled from an poisson distribution
        """
        return np.random.poisson(self.rate_poisson)

    def from_beta(self):
        """
        Sample value from beta distribution.

        Returns
        -------
        x : float
            Value sampled from a beta distribution
        """
        return np.random.beta(self.a, self.b)
    
    def from_be(self):
        """
        Sample value from distribution used to get BE condensation.

        Returns
        -------
        x : float
            Value sampled from distribution used to get BE condensation
        """
        # Calculate the function over the domain (0, 1)
        x = np.linspace(0, 1, 1000)
        y = [self.func_be(float(i)) for i in x]
        
        # Choose value c for the accept reject method
        c = max(y)

        # Accept reject method to sample from given distribution
        while True:
            # Step 1: Simulate the value of Y using the proposal distribution q(n)
            y = self.from_uniform()
            # Step 2: Generate a random number
            u = self.from_uniform()
            # Step 3: Accept or reject the value of Y
            if u <= self.func_be(y) / c:
                # Accept Y and stop
                return y

    def func_be(self, x):
        """
        Calculate the function value at x.

        Parameters
        ----------
        x : float
            x-value for wich the function value should be calculated
        
        Returns
        -------
        f_x : float
            function value at x
        """
        # Check if x is an integer or a float
        if type(x) !=  int and type(x) != float:
            raise TypeError('Invalid type for x, %s. Expected int or float.'%type(x))
        return (1 + self.theta) * (1 - x)**self.theta
    
    def generate_value(self):
        """
        Sample value from current distribution.

        Returns
        -------
        x : float
            Value sampled from current distribution
        """
        if self.current_distribution == 'delta':
            return 1
        else:
            func = getattr(self, 'from_%s'%self.current_distribution)
            return func()
      