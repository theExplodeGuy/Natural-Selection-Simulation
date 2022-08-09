import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations
from random import randint

POPULATION = 50
MAX_BOUNDARY = 5
MIN_BOUNDARY = 0
FOOD_SOURCE = int(POPULATION / 5)


# TODO separate food from organisms

class Organism:
    """A class representing a two-dimensional particle."""

    def __init__(self, x, y, vx, vy, radius=0.01, styles=None):
        """Initialize the particle's position, velocity, and radius.

        An organism is given a randomly generated hex code representing a genome and the attributes belonging to the
        organism.

        Available attributes: size, speed, strength, sense_food, sense_shelter, aggression, kindness, leadership
        """

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius

        # attributes
        self.aggressiveness = randint(1, 15)
        # self.size =
        self.strength = randint(1, 15)
        self.leadership = randint(1, 15)
        self.teamspirit = randint(1, 15)
        # food found
        self.food_found = False
        # food or organism
        self.is_food = False

        self.styles = styles
        if not self.styles:
            # Default circle styles
            self.styles = {'edgecolor': 'b', 'fill': False}

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

        self.init_particles(n, radius, styles)

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

                particle = Organism(x=x, y=y, vx=randint(-30, 30) / 100, vy=randint(-30, 30) / 100, radius=rad, styles=styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break

        for i in range(FOOD_SOURCE):
            # Try to find random available position for organism
            while True:
                # Choose random x,y inside the plane
                styles = {'color': 'C3'}
                x, y = 0.05 + (MAX_BOUNDARY - 2 * 0.05) * np.random.random(2)
                # Choose random velocity
                particle = Organism(x=x, y=y, vx=0, vy=0, radius=0.05, styles=styles)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    particle.is_food = True
                    self.particles.append(particle)
                    break

    def hande_collisions(self):

        def is_food(p1, p2):
            pass


        def is_shelter():
            pass

        def is_organisms():
            pass

        def fight_or_flight():
            pass

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

        """Collisions should be checked amongst all particles. Combinations generates pairs of all Organisms into the 
        self.particles list of Organisms on the fly. """

        pairs = combinations(range(len(self.particles)), 2)
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                move_randomly(self.particles[i], self.particles[j])

    def advance_animation(self, dt):
        """Advance the animation by dt, returning the updated Circles list."""

        for i, p in enumerate(self.particles):
            p.advance(dt)
            self.circles[i].center = p.r
        self.hande_collisions()
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

        self.advance_animation(0.01)
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
            anim.save('simulation.mp4', writer=writer)
        else:
            plt.show()


if __name__ == '__main__':
    n_particles = POPULATION
    radii = np.random.random(n_particles) * 0.05 + 0.02
    styles = {'color': 'C0'}
    sim = Simulation(n_particles, radii, styles)
    sim.do_animation(save=False)
