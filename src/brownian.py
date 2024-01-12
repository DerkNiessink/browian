import numpy as np
from typing import Callable


class BrownianParticle:
    def __init__(self, m: float):
        """
        Initialize a Brownian particle.

        Args:
        - m (float): Mass of the particle.
        """
        self.m = m

    def update_position(
        self, x: float, dt: float, tau: float, T: float, force: Callable
    ):
        """
        Update the position of the Brownian particle.

        Args:
        - x (float): Current position of the particle.
        - dt (float): Time step
        - tau (float): Relaxation time of the system.
        - T (float): Temperature.
        - force (Callable): force on the particle

        Returns:
        - float: New position of the particle after the update.
        """
        chi = np.random.normal()
        f = force(x) if force else 0
        return x + dt * (tau / self.m) * f + np.sqrt(dt * 2 * T * tau / self.m) * chi

    def diffuse(
        self, time: float, dt: float, force: Callable, max_x=None, tau=1.0, T=1.0
    ):
        """
        Evaluate the positions of the Brownian particle over time.

        Args:
        - time (float): Total simulation time.
        - dt (float): Time step.
        - force (Callable): Force that acts on the particle
        - max_x (float | None): Max x for the particle. Simulation is terminated
            when the particle crosses this value. Default is None.
        - tau (float, optional): Relaxation time of the system. Default is 1.
        - T (float, optional): Temperature. Default is 1.

        Returns:
        - tuple: Tuple containing arrays of time and corresponding positions.
        """
        x = 0
        positions = []
        times = np.arange(0, time, dt)

        for _ in times:
            x = self.update_position(x, dt, tau, T, force)
            if max_x and (x > max_x or x < -max_x):
                break
            positions.append(x)

        return times[: len(positions)], positions


class Particles:
    def __init__(self, N: int, force: Callable, T=1.0, tau=1.0, m=1.0):
        """
        Initialize a collection of Brownian particles.

        Args:
        - N (int): Number of particles.
        - force (Callable): Force that acts on the particles
        - T (float, optional): Temperature. Default is 1.
        - tau (float, optional): Relaxation time of the system. Default is 1.
        - m (float, optional): Mass of the particles. Default is 1.
        """
        self.N = N
        self.T = T
        self.tau = tau
        self.force = force
        self.particles = [BrownianParticle(m) for _ in range(N)]

    def diffuse(self, time: float, dt: float, max_x=None):
        """
        Simulate the diffusion of all particles over time.

        Args:
        - time (float): Total simulation time.
        - dt (float): Time step.
        - max_x (float | None): Max x for the particle. Simulation is terminated
            when the particle crosses this value. Default is None.

        Returns:
        - tuple: Tuple containing arrays of time and corresponding positions for
        all particles.
        """
        self.all_positions = []
        self.all_times = []
        for particle in self.particles:
            times, positions = particle.diffuse(
                time, dt, self.force, max_x, self.tau, self.T
            )
            self.all_times.append(np.array(times))
            self.all_positions.append(np.array(positions))
        return self.all_times, self.all_positions

    def mean_square_x(self):
        """
        Calculate the mean square position of all particles.

        Returns:
        - list: List containing the mean square positions over time.
        """
        positions_squared = [x**2 for x in self.all_positions]
        return [np.mean(x) for x in zip(*positions_squared)]
