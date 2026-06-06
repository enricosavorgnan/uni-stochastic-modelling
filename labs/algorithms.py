import numpy as np
import random
from typing import Callable, Tuple, List, Union

class StochasticAlgorithms:
    """
    A collection of core simulation algorithms for stochastic and deterministic systems.
    Includes Discrete-Time Markov Chains (DTMC), Continuous-Time Markov Chains (CTMC),
    Gillespie Stochastic Simulation Algorithm (SSA), SDE solvers (Euler-Maruyama, Milstein),
    Runge-Kutta 4 (RK4), and Splitting methods.
    """

    # =========================================================================
    # 1. DISCRETE-TIME MARKOV CHAINS (DTMC) ALGORITHMS
    # =========================================================================

    @staticmethod
    def dtmc_matrix_project(transition_matrix: np.ndarray, p0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Projects an initial probability distribution vector p0 forward by n_steps
        using the transition matrix of a finite DTMC.
        
        Formula:
            p_n = p_0 * P^n
            where P is the transition matrix.
            
        Details:
            We check that the transition matrix is square and that p0 is a valid
            probability distribution vector of matching dimension.
        """
        size = len(p0)
        if transition_matrix.shape != (size, size):
            raise ValueError("Transition matrix dimensions must match initial state vector.")
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if not np.isclose(p0.sum(), 1.0) or np.any(p0 < 0):
            raise ValueError("p0 must be a valid probability distribution vector.")

        # Compute P^n by iterative matrix multiplication
        mat_pow = transition_matrix.copy()
        for _ in range(1, n_steps):
            mat_pow = np.matmul(mat_pow, transition_matrix)

        return np.dot(p0, mat_pow)

    @staticmethod
    def dtmc_matrix_trajectory(transition_matrix: np.ndarray, p0: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Computes the probability distribution at each step of a finite DTMC
        from step 0 to n_steps.
        
        Returns:
            A 2D array of shape (size, n_steps + 1) where each column represents
            the probability distribution vector at that time step.
        """
        size = len(p0)
        if transition_matrix.shape != (size, size):
            raise ValueError("Transition matrix dimensions must match initial state vector.")
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if not np.isclose(p0.sum(), 1.0) or np.any(p0 < 0):
            raise ValueError("p0 must be a valid probability distribution vector.")

        trajectory = np.zeros((size, n_steps + 1), dtype=float)
        trajectory[:, 0] = p0

        mat_pow = transition_matrix.copy()
        for step in range(1, n_steps):
            trajectory[:, step] = np.dot(p0, mat_pow)
            mat_pow = np.matmul(mat_pow, transition_matrix)
        trajectory[:, -1] = np.dot(p0, mat_pow)

        return trajectory

    @staticmethod
    def dtmc_sample_state(p0: np.ndarray) -> int:
        """
        Helper method to sample a discrete state index from a probability distribution.
        """
        cdf = np.cumsum(p0)
        r = random.uniform(0, 1)
        for state_idx, val in enumerate(cdf):
            if r <= val:
                return state_idx
        raise RuntimeError("CDF sampling failed. Ensure probability vector sums to 1.0.")

    @staticmethod
    def dtmc_sample_trajectory(n_steps: int, p0: np.ndarray, 
                               transition_prob_fn: Callable[[int], Tuple[float, float, float]]) -> np.ndarray:
        """
        Simulates a single sample trajectory of a DTMC (possibly with infinite states).
        At each step, transition probabilities (up, down, stay) are evaluated by a function
        taking the current state as input.
        
        transition_prob_fn(state) -> (p_up, p_down, p_stay)
        """
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if not np.isclose(p0.sum(), 1.0) or np.any(p0 < 0):
            raise ValueError("p0 must be a valid probability distribution vector.")

        # Sample initial state index
        state_trajectory = np.zeros(n_steps + 1, dtype=int)
        state_trajectory[0] = StochasticAlgorithms.dtmc_sample_state(p0)

        for step in range(1, n_steps + 1):
            curr_state = state_trajectory[step - 1]
            p_up, p_down, p_stay = transition_prob_fn(curr_state)
            
            # Normalize to handle numerical rounding issues
            total = p_up + p_down + p_stay
            if total <= 0:
                # Absorbent or dead end state, remain the same
                state_trajectory[step] = curr_state
                continue
            
            p_up /= total
            p_down /= total
            
            r = random.uniform(0, 1)
            if r < p_up:
                state_trajectory[step] = curr_state + 1
            elif r < p_up + p_down:
                state_trajectory[step] = curr_state - 1
            else:
                state_trajectory[step] = curr_state

        return state_trajectory

    # =========================================================================
    # 2. CONTINUOUS-TIME MARKOV CHAINS (CTMC) & GILLESPIE SSA
    # =========================================================================

    @staticmethod
    def ctmc_kolmogorov_solve(generator_matrix: np.ndarray, p0: np.ndarray, T: float, M: int) -> np.ndarray:
        """
        Solves the Forward Kolmogorov Equation: dP(t)/dt = P(t) * Q
        using a custom Runge-Kutta 4 system solver (avoiding dependency on scipy.integrate.odeint).
        
        Args:
            generator_matrix (Q): Infinitesimal generator of shape (S, S)
            p0: Initial probability distribution of shape (S,)
            T: Total time
            M: Number of time step partition points
            
        Returns:
            A 2D array of shape (M, S) containing the probability distribution at each time step.
        """
        num_states = len(p0)
        t_eval = np.linspace(0, T, M)
        dt = T / (M - 1)
        
        sol = np.zeros((M, num_states), dtype=float)
        sol[0, :] = p0
        
        # System function: dy/dt = y * Q
        def forward_eq(y):
            return np.dot(y, generator_matrix)
        
        # RK4 Integration
        for step in range(1, M):
            y_curr = sol[step - 1, :]
            
            k1 = forward_eq(y_curr)
            k2 = forward_eq(y_curr + 0.5 * dt * k1)
            k3 = forward_eq(y_curr + 0.5 * dt * k2)
            k4 = forward_eq(y_curr + dt * k3)
            
            sol[step, :] = y_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            
        return sol

    @staticmethod
    def gillespie_ssa_finite(generator_matrix: np.ndarray, p0: np.ndarray, T: float) -> int:
        """
        Simulates a single trajectory of a finite CTMC up to time T using the Gillespie SSA algorithm.
        
        Steps:
            1. Construct the transition matrix of the embedded Jump Chain J:
               J_ij = Q_ij / -Q_ii for i != j, J_ii = 0.
            2. At each state, sample the next state from the row J[state, :]
            3. Sample the transition time from an Exponential distribution with rate -Q[state, state]
            4. Repeat until accumulated time exceeds T.
        """
        # Build jump matrix J
        n = generator_matrix.shape[0]
        J = np.zeros((n, n), dtype=float)
        for i in range(n):
            diagonal_val = -generator_matrix[i, i]
            if diagonal_val > 1e-12:
                J[i, :] = generator_matrix[i, :] / diagonal_val
                J[i, i] = 0.0
            else:
                # Absorbent state (Q_ii == 0)
                J[i, i] = 1.0

        # Sample initial state
        curr_state = StochasticAlgorithms.dtmc_sample_state(p0)
        curr_time = 0.0

        while curr_time < T:
            # If state is absorbent, time jumps to T
            rate = -generator_matrix[curr_state, curr_state]
            if rate < 1e-12:
                break
                
            # Sample next state from jump chain
            next_state = StochasticAlgorithms.dtmc_sample_state(J[curr_state, :])
            
            # Sample exponential time interval: delta_t = -log(1 - r) / rate
            r = random.random()
            delta_t = -np.log(1.0 - r) / rate
            curr_time += delta_t
            
            if curr_time < T:
                curr_state = next_state

        return curr_state

    @staticmethod
    def gillespie_ssa_population(birth_rate_fn: Callable[[int], float], 
                                 death_rate_fn: Callable[[int], float], 
                                 p0: np.ndarray, T: float) -> int:
        """
        Simulates a population CTMC trajectory (e.g. Malthusian birth-death) up to time T.
        
        Args:
            birth_rate_fn: Function returning the birth rate given population size N.
            death_rate_fn: Function returning the death rate given population size N.
            p0: Probability distribution for the initial population size.
            T: End time.
            
        Returns:
            The final population size.
        """
        curr_state = StochasticAlgorithms.dtmc_sample_state(p0)
        curr_time = 0.0

        while curr_time < T:
            b_rate = birth_rate_fn(curr_state)
            d_rate = death_rate_fn(curr_state)
            total_rate = b_rate + d_rate

            if total_rate < 1e-12:
                break # Population extinct or system absorbed

            # Sample time step
            r1 = random.random()
            delta_t = -np.log(1.0 - r1) / total_rate
            curr_time += delta_t

            if curr_time < T:
                # Sample transition: Birth vs Death
                r2 = random.random()
                prob_birth = b_rate / total_rate
                if r2 < prob_birth or curr_state <= 0:
                    curr_state += 1
                else:
                    curr_state -= 1

        return curr_state

    @staticmethod
    def gillespie_ssa_crn(initial_state: np.ndarray, 
                          propensity_fn: Callable[[np.ndarray], np.ndarray], 
                          stoichiometry_matrix: np.ndarray, 
                          T: float) -> np.ndarray:
        """
        Simulates a Chemical Reaction Network (CRN) trajectory using Gillespie SSA.
        
        Args:
            initial_state: Array of chemical quantities/populations of shape (N_species,)
            propensity_fn: Function returning the propensity of each reaction given state. Shape (N_reactions,)
            stoichiometry_matrix: Matrix of shape (N_species, N_reactions) where column j is the net change vector of reaction j.
            T: End time of the simulation.
            
        Returns:
            A 2D array of shape (N_time_steps, N_species) representing the trajectory.
        """
        states = [initial_state.astype(float).copy()]
        state = initial_state.astype(float).copy()
        curr_time = 0.0

        while curr_time < T:
            propensities = propensity_fn(state)
            total_propensity = np.sum(propensities)

            if total_propensity < 1e-12:
                break # No further reactions can occur

            # Sample transition time (exponential distribution)
            tau = np.random.exponential(1.0 / total_propensity)
            curr_time += tau
            
            if curr_time > T:
                break

            # Choose which reaction occurs based on propensity weights
            r = random.random()
            cdf_prop = np.cumsum(propensities) / total_propensity
            chosen_reaction = 0
            for idx, val in enumerate(cdf_prop):
                if r < val:
                    chosen_reaction = idx
                    break

            # Update state vector according to chosen reaction stoichiometry
            state += stoichiometry_matrix[:, chosen_reaction]
            states.append(state.copy())

        return np.array(states)

    # =========================================================================
    # 3. STOCHASTIC DIFFERENTIAL EQUATIONS (SDE) SOLVERS
    # =========================================================================

    @staticmethod
    def euler_deterministic(x0: float, drift_fn: Callable[[float], float], 
                            t_max: float, dt: float) -> np.ndarray:
        """
        Euler integration scheme for deterministic ODEs:
            dx/dt = f(x)
            x_{n+1} = x_n + dt * f(x_n)
            
        Cumulative error order: O(dt)
        """
        n_steps = int(round(t_max / dt))
        traj = np.zeros(n_steps + 1, dtype=float)
        traj[0] = x0

        for i in range(1, n_steps + 1):
            traj[i] = traj[i - 1] + dt * drift_fn(traj[i - 1])

        return traj

    @staticmethod
    def euler_maruyama(x0: float, drift_fn: Callable[[float], float], 
                       diffusion_fn: Callable[[float], float], 
                       t_max: float, dt: float) -> np.ndarray:
        """
        Euler-Maruyama integration scheme for stochastic SDEs in Ito form:
            dx = f(x)dt + g(x)dW
            x_{n+1} = x_n + dt * f(x_n) + g(x_n) * sqrt(dt) * Z
            where Z ~ N(0, 1)
            
        Cumulative error order: O(sqrt(dt)) (strongly converges at this order)
        """
        n_steps = int(round(t_max / dt))
        traj = np.zeros(n_steps + 1, dtype=float)
        traj[0] = x0

        for i in range(1, n_steps + 1):
            x_prev = traj[i - 1]
            drift = drift_fn(x_prev)
            diffusion = diffusion_fn(x_prev)
            Z = random.gauss(0, 1)
            
            traj[i] = x_prev + dt * drift + diffusion * np.sqrt(dt) * Z

        return traj

    @staticmethod
    def milstein(x0: float, drift_fn: Callable[[float], float], 
                 diffusion_fn: Callable[[float], float], 
                 diffusion_prime_fn: Callable[[float], float], 
                 t_max: float, dt: float) -> np.ndarray:
        """
        Milstein integration scheme for SDEs in Ito form:
            dx = f(x)dt + g(x)dW
            x_{n+1} = x_n + dt * f(x_n) + g(x_n) * sqrt(dt) * Z + 0.5 * g(x_n) * g'(x_n) * dt * (Z^2 - 1)
            where Z ~ N(0, 1)
            
        Cumulative error order: O(dt)
        Requires the derivative of the diffusion term g'(x).
        """
        n_steps = int(round(t_max / dt))
        traj = np.zeros(n_steps + 1, dtype=float)
        traj[0] = x0

        for i in range(1, n_steps + 1):
            x_prev = traj[i - 1]
            drift = drift_fn(x_prev)
            diffusion = diffusion_fn(x_prev)
            diff_prime = diffusion_prime_fn(x_prev)
            Z = random.gauss(0, 1)

            traj[i] = (x_prev + 
                       dt * drift + 
                       diffusion * np.sqrt(dt) * Z + 
                       0.5 * diffusion * diff_prime * dt * (Z * Z - 1.0))

        return traj

    @staticmethod
    def rk4_deterministic(x0: Union[float, np.ndarray], 
                          drift_fn: Callable[[float, Union[float, np.ndarray]], Union[float, np.ndarray]], 
                          t_max: float, dt: float) -> np.ndarray:
        """
        Runge-Kutta 4th order solver for deterministic ODEs.
        Handles both scalar states and 1D vector states.
        
        Cumulative error order: O(dt^4)
        """
        n_steps = int(round(t_max / dt))
        
        if isinstance(x0, np.ndarray):
            traj = np.zeros((n_steps + 1, len(x0)), dtype=float)
        else:
            traj = np.zeros(n_steps + 1, dtype=float)
            
        traj[0] = x0

        for i in range(1, n_steps + 1):
            t_curr = (i - 1) * dt
            y_curr = traj[i - 1]

            k1 = drift_fn(t_curr, y_curr)
            k2 = drift_fn(t_curr + 0.5 * dt, y_curr + 0.5 * dt * k1)
            k3 = drift_fn(t_curr + 0.5 * dt, y_curr + 0.5 * dt * k2)
            k4 = drift_fn(t_curr + dt, y_curr + dt * k3)

            traj[i] = y_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return traj

    @staticmethod
    def splitting_rk4_milstein(x0: float, drift_fn: Callable[[float], float], 
                               diffusion_fn: Callable[[float], float], 
                               diffusion_prime_fn: Callable[[float], float], 
                               t_max: float, dt: float) -> np.ndarray:
        """
        A naive splitting scheme for SDEs:
            Deterministic drift is integrated using RK4 (error order O(dt^4))
            Stochastic diffusion is integrated using Milstein (error order O(dt))
            
        Formula:
            x_{n+1} = x_n + RK4_step(x_n, dt) + Milstein_stochastic_step(x_n, dt)
            
        This significantly reduces the deterministic component's contribution to cumulative error.
        """
        n_steps = int(round(t_max / dt))
        traj = np.zeros(n_steps + 1, dtype=float)
        traj[0] = x0

        for i in range(1, n_steps + 1):
            x_prev = traj[i - 1]
            
            # 1. Deterministic update via RK4
            k1 = drift_fn(x_prev)
            k2 = drift_fn(x_prev + 0.5 * dt * k1)
            k3 = drift_fn(x_prev + 0.5 * dt * k2)
            k4 = drift_fn(x_prev + dt * k3)
            det_change = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # 2. Stochastic update via Milstein
            diffusion = diffusion_fn(x_prev)
            diff_prime = diffusion_prime_fn(x_prev)
            Z = random.gauss(0, 1)
            stoch_change = (diffusion * np.sqrt(dt) * Z + 
                            0.5 * diffusion * diff_prime * dt * (Z * Z - 1.0))

            traj[i] = x_prev + det_change + stoch_change

        return traj
