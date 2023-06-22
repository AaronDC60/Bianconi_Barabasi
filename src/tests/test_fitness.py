"""Test file for generating values from different distributions."""

# Imports
import numpy as np
import pytest
from src import fitness

def test_set_parameters():
    """
    Check if setting all the parameters works properly.
    """
    generator = fitness.generator()

    # Expected errors (no integer or float)
    with pytest.raises(TypeError):
        generator.set_param_exp('invalid type')
    with pytest.raises(TypeError):
        generator.set_param_poisson('invalid type')
    with pytest.raises(TypeError):
        generator.set_param_beta('invalid type', 1.5)
    with pytest.raises(TypeError):
        generator.set_param_beta(1.5, 'invalid type')
    with pytest.raises(TypeError):
        generator.set_param_be('invalid type')
    
    # Expected errors (negative values) 
    with pytest.raises(ValueError):
        generator.set_param_exp(-1.5)
    with pytest.raises(ValueError):
        generator.set_param_poisson(-1.5)
    with pytest.raises(ValueError):
        generator.set_param_beta(-1.5, 1.5)
    with pytest.raises(ValueError):
        generator.set_param_beta(1.5, -1.5)
    with pytest.raises(ValueError):
        generator.set_param_be(-1.5)

    generator.set_param_exp(3)
    generator.set_param_poisson(4)
    generator.set_param_beta(5,6)
    generator.set_param_be(7)

    assert(generator.rate_exp == 3)
    assert(generator.rate_poisson == 4)
    assert(generator.a == 5)
    assert(generator.b == 6)
    assert(generator.theta == 7)

def test_current_distr():
    """
    Check if setting and generating from a distribution works.
    """
    generator = fitness.generator()

    # Check default (delta)
    assert(generator.generate_value() == 1)

    # Use unkown distribution
    with pytest.raises(NameError):
        generator.set_current_distribution(1.5)
    with pytest.raises(NameError):
        generator.set_current_distribution('distribution x')
    
    # Check for every distribution of the correct one is being sampled from
    generator.set_current_distribution('uniform')
    np.random.seed(1)
    x1 = generator.generate_value()
    np.random.seed(1)
    x2 = generator.from_uniform()
    assert(x1 == x2)

    generator.set_current_distribution('exponential')
    np.random.seed(1)
    x1 = generator.generate_value()
    np.random.seed(1)
    x2 = generator.from_exponential()
    assert(x1 == x2)

    generator.set_current_distribution('poisson')
    np.random.seed(1)
    x1 = generator.generate_value()
    np.random.seed(1)
    x2 = generator.from_poisson()
    assert(x1 == x2)

    generator.set_current_distribution('beta')
    np.random.seed(1)
    x1 = generator.generate_value()
    np.random.seed(1)
    x2 = generator.from_beta()
    assert(x1 == x2)

    generator.set_current_distribution('be')
    np.random.seed(1)
    x1 = generator.generate_value()
    np.random.seed(1)
    x2 = generator.from_be()
    assert(x1 == x2)
