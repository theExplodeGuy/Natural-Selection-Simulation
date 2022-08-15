import sys
import time
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from random import randint

POPULATION = 20
MAX_BOUNDARY = 5
MIN_BOUNDARY = 0
FOOD_SOURCE = 15
ENERGY = 3000000 / 2


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
        self.team_spirit = randint(1, 15)

        # food found
        self.food_found = False
        # food or organism
        self.is_food = False
        self.is_fed = False
        self.has_offspring = False

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
            self.energy -= 200 * self.radius + (1000 / 41) * (sqrt(self.vx ** 2 + self.vy ** 2)) + (
                    2000 / 14) * self.strength

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

                    color = int(p1.vx + p2.vy + ((p2.radius + p1.radius) / 2))

                    styles = {'color': f'C{color}', 'fill': False}

                    # Check that the Particle doesn't overlap one that's already
                    # been placed.
                    particle = Organism(x=x, y=y, vx=p1.vx, vy=p2.vy, radius=(p2.radius + p1.radius) / 2,
                                        styles=styles)

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
            """else:
                co_operate(p1, p2)"""

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

        def co_operate():
            pass

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

            elif p.energy < 1:
                self.particles.remove(p)
                print('removed', i)

            elif p not in self.food_to_remove and p.energy > 1:
                self.circles[i].center = p.r

        self.init()
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

    def do_animation(self, save=False):
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
        else:
            plt.show()


if __name__ == '__main__':
    n_particles = POPULATION
    radii = np.random.random(n_particles) * 0.05 + 0.02
    # styles = {'color': 'C0'}
    sim = Simulation(n_particles, radii)
    sim.do_animation(save=False)
