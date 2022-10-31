from main import Organism, Simulation
import pytest

def test_energy():
    organism = Organism(0, 0, 1, 1)
    assert organism.energy == 2000000

def test_speed():
    organism = Organism(0, 0, 3, 4)
    assert organism.speed == 5

def test_range():
    org = Organism(0, 0, 3, 4)
    assert org.strength < 16 and org.strength > 0
    assert org.aggressiveness < 16 and org.aggressiveness > 0
    assert org.consumption < 2267 and org.consumption > 265












