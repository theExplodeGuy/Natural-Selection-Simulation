import random
import sys
import time
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from random import randint
from utilities import plot_agg_attribute
from utilities import plot_power_attribute
from utilities import plot_cooperation_attribute
from utilities import collect_data


POPULATION = 1000
MAX_BOUNDARY = 15
MIN_BOUNDARY = 0
FOOD_SOURCE = 200
ENERGY = 3000000 / 2.5


# TODO separate food from organisms

class Organism:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        """Initialize the particle's position, velocity, and radius.

        An organism is given a randomly generated hex code representing a genome and the attributes belonging to the
        organism.

        Available attributes: size, speed, strength, sense_food, sense_shelter, aggression, kindness, leadership
        """
        # inital energy
        self.energy = ENERGY

        # initial position
        self.r = np.array((x, y))

        self.v = np.array((vx, vy))
        # size
        self.radius = radius

        # attributes
        self.aggressiveness = randint(1, 15)

        self.strength = randint(1, 15)
        self.leadership = randint(1, 15)
        self.team_spirit = 15 - self.aggressiveness
        self.speed = sqrt(self.vx ** 2 + self.vy ** 2)

        # food found
        self.food_found = False
        # food or organism
        self.is_food = False
        self.is_fed = False
        self.has_offspring = False
        self.has_cooperated = 0

        # score for fight or flight
        if (4 * self.aggressiveness + 2 * self.strength + 280 * self.radius) - 7 * (1400 / 41) * self.speed > 9.2:
            self.score = True
        else:
            self.score = False

        self.power = 2 * self.strength + 280 * self.radius
        self.consumption = 200 * self.radius + (1000 / 41) * self.speed + (2000 / 14) * self.strength

        self.color = int(
            self.aggressiveness + self.strength + self.leadership + self.team_spirit + self.radius + self.vx \
            + self.vy)
        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'color': f'C{self.color}', 'fill': True}

    # For convenience, map the components of the particle's position and
    # velocity vector onto the attributes x, y, vx and vy.

    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value

    @property
    def vx(self):
        return self.v[0]

    @vx.setter
    def vx(self, value):
        self.v[0] = value

    @property
    def vy(self):
        return self.v[1]

    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.r += self.v * dt

        if not self.is_food:
            self.energy -= self.consumption

        # Make the Particles bounce off the walls
        if self.x - self.radius < MIN_BOUNDARY:
            self.x = self.radius
            self.vx = -self.vx
        if self.x + self.radius > MAX_BOUNDARY:
            self.x = MAX_BOUNDARY - self.radius
            self.vx = -self.vx
        if self.y - self.radius < MIN_BOUNDARY:
            self.y = self.radius
            self.vy = -self.vy
        if self.y + self.radius > MAX_BOUNDARY:
            self.y = MAX_BOUNDARY - self.radius
            self.vy = -self.vy


class Simulation:
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: MIN_BOUNDARY <= x < MAX_BOUNDARY, MIN_BOUNDARY <= y < MAX_BOUNDARY.

    """

    def __init__(self, n, radius=0.01, styles=None):
        """Initialize the simulation with n Particles with radii radius.

        radius can be a single value or a sequence with n values.
        """

        self.max_attribute = []
        self.avg_agg_attribute = [0]
        self.avg_power_attribute = [0]
        self.avg_cooperation_attribute = [0]
        self.organism_to_remove = set()
        self.circles = []
        self.to_add = []
        self.food_to_remove = set()
        self.total_food = set()
        self.init_particles(n, radius, styles)
        self.init_food()

    def init_particles(self, n, radius, styles=None):
        """Initialize the n Particles of the simulation.

        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.
        TODO Velocity and Radius should be decided according to genome not randomly
        """

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius

            radius = r_gen(n, radius)

        self.n = n
        self.particles = []

        for i, rad in enumerate(radius):
            # Try to find random available position for organism
            while True:
                # Choose random x,y inside the plane
                x, y = rad + (MAX_BOUNDARY - 2 * rad) * np.random.random(2)
                # Choose random velocity

                particle = Organism(x=x, y=y, vx=randint(-30, 30) / 100, vy=randint(-30, 30) / 100, radius=rad,
                                    styles=styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

    def init_food(self):
        for i in range(FOOD_SOURCE):
            # Try to find random available position for organism
            while True:
                # Choose random x,y inside the plane
                styles = {'color': 'C7', 'fill': True}
                x, y = 0.05 + (MAX_BOUNDARY - 2 * 0.05) * np.random.random(2)
                # Choose random velocity
                particle = Organism(x=x, y=y, vx=0, vy=0, radius=0.1, styles=styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    particle.is_food = True
                    self.particles.append(particle)
                    self.total_food.add(particle)
                    break

    def hande_collisions(self):

        def is_food(p1, p2):
            if p1.is_food and not p2.is_food:
                p2.energy = ENERGY
                p2.is_fed = True
                p1.vx = 0
                p1.vy = 0
                return p1
            elif not p1.is_food and p2.is_food:
                p1.energy = ENERGY
                p1.is_fed = True
                p2.vx = 0
                p2.vy = 0
                return p2

        def reproduce(p1, p2):
            if p1.is_fed and p2.is_fed and not p1.has_offspring and not p2.has_offspring:
                # p1.has_offspring = True
                # p2.has_offspring = True
                while True:
                    x, y = 0.05 + (MAX_BOUNDARY - 2 * 0.05) * np.random.random(2)

                    color = int(p1.vx + p2.vy + ((p2.radius + p1.radius) / 2)
                                + (p1.aggressiveness + p2.aggressiveness) / 2 + (p1.strength + p2.strength) / 2)

                    styles = {'color': f'C{color}', 'fill': False}

                    # Check that the Particle doesn't overlap one that's already
                    # been placed.
                    particle = Organism(x=x, y=y, vx=p1.vx, vy=p2.vy, radius=(p2.radius + p1.radius) / 2,
                                        styles=styles)

                    particle.strength = (p1.strength + p2.strength) / 2
                    particle.aggressiveness = (p1.aggressiveness + p2.aggressiveness) / 2
                    particle.radius = (p1.radius + p2.radius)/2
                    particle.leadership = (p1.leadership + p2.leadership)/2
                    particle.vx = (p1.vx + p2.vx)/2
                    particle.vy = (p1.vy + p2.vy)/2
                    update_organism(particle)

                    for p2 in self.particles:
                        if p2.overlaps(particle):
                            break
                    else:
                        self.particles.append(particle)
                        self.init()
                        print('born')
                        break

        def is_organisms(p1, p2):
            if p1.score and p2.score:
                return fight_result(p1, p2)
            elif p1.score and not p2.score:
                return fight_vs_flight(p1, p2)
            elif p2.score and not p1.score:
                return fight_vs_flight(p2, p1)
            else:
                return cooperate(p1, p2)

        def fight_result(p1, p2):
            if p1.power > p2.power and p1.consumption < p1.energy:
                p1.energy = ENERGY
                return p2
            if p2.power > p1.power and p2.consumption < p2.energy:
                p2.energy = ENERGY
                return p1

        def fight_vs_flight(fighter, runner):
            if fighter.speed > runner.speed:
                fighter.energy = ENERGY
                return runner
            elif runner.speed > fighter.speed:
                return False
            else:
                fight_result(fighter, runner)

        def cooperate(p1, p2):
            # check if leadership is compatible
            if p1.leadership == p2.leadership:
                return
            # check which is the leader
            elif p1.leadership > p2.leadership:
                # setting new attributes to the leader
                p1.radius = p1.radius + 0.2 * p2.radius
                p1.aggressiveness = p1.aggressiveness + 0.2 * p2.aggressiveness
                p1.strength = p1.strength + 0.2 * p2.strength
                p1.leadership = p1.leadership + 0.2 * p2.leadership
                # comparing speeds to set it to the on of the slower organism
                if p1.vx > p2.vx:
                    p1.vx = p2.vx

                if p1.vy > p2.vy:
                    p1.vy = p2.vy
                update_organism(p1)
                p1.has_cooperated += 1
                return p2
            else:
                # setting new attributes to leader
                p2.radius = p2.radius + 0.2 * p1.radius
                p2.aggressiveness = p2.aggressiveness + 0.2 * p1.aggressiveness
                p2.strength = p2.strength + 0.2 * p1.strength
                p2.leadership = p2.leadership + 0.2 * p1.leadership
                # comparing speeds to set it to the on of the slower organism
                if p2.vx > p1.vx:
                    p2.vx = p1.vx

                if p2.vy > p1.vy:
                    p2.vy = p1.vy
                update_organism(p2)
                p2.has_cooperated += 1
                return p1

        def update_organism(p1):
            p1.speed = sqrt(p1.vx ** 2 + p1.vy ** 2)
            if (4 * p1.aggressiveness + 2 * p1.strength + 280 * p1.radius) - 7 * (1400 / 41) * p1.speed > 9.2:
                p1.score = True
            else:
                p1.score = False
            p1.power = 2 * p1.strength + 280 * p1.radius
            p1.consumption = 200 * p1.radius + (1000 / 41) * p1.speed + (2000 / 14) * p1.strength

        def split_food():
            pass

        # move randomly after collision according to these equations:
        # https://en.wikipedia.org/wiki/Elastic_collision
        def move_randomly(p1, p2):
            m1, m2 = p1.radius ** 2, p2.radius ** 2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2) ** 2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2 * m2 / M * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
            u2 = v2 - 2 * m1 / M * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2

        def remove_pair(pair_list, x):
            pair_list.remove(x)

        """Collisions should be checked amongst all particles. Combinations generates pairs of all Organisms into the 
        self.particles list of Organisms on the fly. """
        pairs = list(combinations(range(len(self.particles)), 2))
        for i, j in pairs[:]:
            if self.particles[i].overlaps(self.particles[j]):

                food = is_food(self.particles[i], self.particles[j])
                if food:
                    self.food_to_remove.add(food)

                elif self.particles[i].is_fed and self.particles[j].is_fed:
                    reproduce(self.particles[i], self.particles[j])

                elif is_organisms(self.particles[i], self.particles[j]):
                    self.organism_to_remove.add(is_organisms(self.particles[i], self.particles[j]))

                move_randomly(self.particles[i], self.particles[j])

    def spawn_food(self):
        if not self.total_food:
            self.init_food()
            self.init()



    def advance_animation(self, dt):
        """Advance the animation by dt, returning the updated Circles list."""
        self.hande_collisions()
        self.spawn_food()
        if len(self.particles) == 0:
            sys.exit('END')

        for i, p in enumerate(self.particles):
            p.advance(dt)
            if p in self.food_to_remove:
                self.food_to_remove.remove(p)
                self.total_food.remove(p)
                self.particles.remove(p)
            elif p in self.organism_to_remove:
                self.organism_to_remove.remove(p)
                self.particles.remove(p)
                print('removed by fight/flight', i)

            elif p.energy < 1:
                self.particles.remove(p)
                print('removed by energy', i)

            elif p not in self.food_to_remove and p.energy > 1:
                self.circles[i].center = p.r

        self.init()
        collect_data(self)
        return self.circles

    def advance(self, dt):
        """Advance the animation by dt."""
        for i, p in enumerate(self.particles):
            p.advance(dt)
        self.hande_collisions()

    def init(self):
        """Initialize Matplotlib animation."""

        self.circles = []

        for particle in self.particles:
            self.circles.append(particle.draw(self.ax))

        return self.circles

    def animate(self, i):
        """The function passed to Matplotlib FuncAnimation routine"""

        self.advance_animation(0.08)

        return self.circles

    def do_animation(self, save=False, graphs=True):
        """Set up and carry animation.
        To save animation as a MP4 movie, set save=True."""

        fig, self.ax = plt.subplots()
        for s in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(MIN_BOUNDARY, MAX_BOUNDARY)
        self.ax.set_ylim(MIN_BOUNDARY, MAX_BOUNDARY)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        anim = animation.FuncAnimation(fig, self.animate, init_func=self.init, frames=800, interval=2, blit=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=100, bitrate=1800)
            anim.save('simulationv2.mp4', writer=writer)
        if graphs:
            plt.show()
            plot_agg_attribute(self.avg_agg_attribute)
            plt.show()
            plot_power_attribute(self.avg_power_attribute)
            plt.show()
            plot_cooperation_attribute(self.avg_cooperation_attribute)

        else:
            plt.show()


if __name__ == '__main__':
    n_particles = POPULATION
    radii = np.random.random(n_particles) * 0.05 + 0.02
    # styles = {'color': 'C0'}
    sim = Simulation(n_particles, radii)
    sim.do_animation(save=False)
