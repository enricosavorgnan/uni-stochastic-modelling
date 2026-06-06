import time
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable


class WienerProcess:
    """
    Provide tools for simulating a Wiener process.
    """

    def run(self, w_0 : int | float = 0. , steps : int = 100, dt = 1.):
        """
        Run a Wiener Process simulation.

        Parameters:
        -----------
        - w_0 : int | float, default: 0
            Initial value of the Wiener process
        - steps : int, default: 100
            Number of steps to simulate

        Returns:
        --------
        sequence : np.ndarray
            Sequence of Wiener steps
        """
        sequence = np.zeros(steps+1)
        sequence[0] = w_0

        for t in range(1, steps+1):
            sequence[t] = sequence[t-1] + np.random.normal(loc = 0, scale = np.sqrt(dt))

        return sequence


class Ito:
    """
    Provide tools for the calculus of the Ito Integral.
    """

    def integral(self,
                 X: Callable,
                 W : list[float | int] | np.ndarray | None = None,
                 steps: int | None = None,
                 t_0 : int | float | None = 0.,
                 t_n : int | float | None = 1.,
                 plot : bool = False):

        """
        Compute the Ito integral of a function x with respect to a Wiener process w.

        Parameters:
        -----------
        - X : list[float | int ] | np.ndarray | Callable
            Function to integrate. Can be a list of values, a numpy array, or a callable function.
        - W : list[float | int] | np.ndarray | None, default: None
            Sequence of Wiener process values. Must be the same length as x if x is a list or numpy array.
            If not provided, a Wiener process with the same length as X will be simulated starting from w_0 = 0.
        - steps : int | None, default: None
            Number of steps to simulate if w is not provided.
            Must be provided if w is None and x is a callable function.
        - t_0 : int | float | None, default: 0
            Initial time. Must be provided if x is a callable function.
        - t_n : int | float | None, default: 1
            Final time. Must be provided if x is a callable function.
        - plot : bool, default: False
            Whether to plot the function x, the Wiener process w, and the Ito integral.
        """
        assert steps is not None, "If x is a callable function, steps must be provided."
        times = np.linspace(t_0, t_n, steps+1)
        dt = times[1] - times[0]

        x = X(times)
        w = WienerProcess().run(w_0=0, steps = steps, dt = dt)
        ito_integral, ito_point_wise = self.compute_ito_integral(x, w)

        if plot:
            self.plot(x, w, ito_point_wise)

        return x, w, ito_point_wise, ito_integral,


    def compute_ito_integral(self, x: np.ndarray, w: np.ndarray):
        """
        Compute the Ito integral using the left Riemann sum approximation.

        Parameters:
        -----------
        - x : np.ndarray
            Sequence of function values at the left endpoints of the intervals.
        - w : np.ndarray
            Sequence of Wiener process values at the left endpoints of the intervals.

        Returns:
        --------
        integral : float
            Approximation of the Ito integral.
        """
        print(len(x), len(w))
        assert len(x) == len(w), "Xs and ws must have the same length."

        integral = 0.
        integral_point_wise = np.zeros(len(x))

        for i in range(1, len(x)):
            integral += x[i-1] * (w[i] - w[i-1])
            integral_point_wise[i] = integral

        return integral, integral_point_wise


    def plot(self, x, w, i):
        """
        Plot the function x, the Wiener process w, and the Ito integral i.

        Parameters:
        -----------
        - x : np.ndarray
            Sequence of function values.
        - w : np.ndarray
            Sequence of Wiener process values.
        - i : float
            Value of the Ito integral.

        Returns:
        --------
        None
        """
        # fig = plt.figure(figsize=(16, 9), dpi=600)
        # plt.subplot(2, 1, 1)
        # plt.plot(x, label='x(t)')
        # plt.plot(w, label='W(t)', color='orange', alpha=0.5)
        # plt.plot(x+w, label='Function', color='purple', alpha=0.5)
        #
        # plt.plot(i, label='Ito Integral', color='green')
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.show()

        # Plot all in one graph
        plt.figure(figsize=(16, 9), dpi=600)
        plt.plot(x, label='x(t)')
        plt.plot(w, label='W(t)', color='orange', alpha=0.5)
        plt.plot(x+w, label='Function', color='purple', alpha=0.5)
        plt.plot(i, label='Ito Integral', color='green')
        plt.legend()
        plt.title('Ito Integral Simulation')

        # Save
        plt.savefig(f'ito_integral_{int(time.time())}.png')
        plt.show()



if __name__ == "__main__":
    # Example usage
    ito = Ito()
    steps = 1000
    t_0 = 0
    t_n = 30

    # x = lambda t : np.sin(2 * np.pi * t)
    x = lambda t : 2 * t**1.5 + 12 * np.sin(t) - 6 * t

    values, wiener, integral_pw, integral = ito.integral(X=x, steps=steps, t_0=t_0, t_n=t_n, plot = True)
    print(f"Ito integral: {integral}")