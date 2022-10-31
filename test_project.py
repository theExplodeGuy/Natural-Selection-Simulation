from main import Organism


def test_energy():
    organism = Organism(0, 0, 1, 1)
    assert organism.energy == 2000000


def test_speed():
    organism = Organism(0, 0, 3, 4)
    assert organism.speed == 5


def test_range():
    org = Organism(0, 0, 3, 4)
    assert 16 > org.strength > 0
    assert 16 > org.aggressiveness > 0
    assert 2267 > org.consumption > 265
