import pytest
from sklearn.datasets import make_moons

@pytest.fixture
def dataset():
    return make_moons(n_samples=50, noise=0.2, random_state=43)