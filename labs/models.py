import numpy as np
import random
import copy
from typing import List, Tuple, Union
from algorithms import StochasticAlgorithms

# =========================================================================
# 1. DISCRETE-TIME MARKOV CHAINS (DTMC) MODELS
# =========================================================================

class OSMalth:
    """Oversimplified Malthusian System (Finite State DTMC via Transition Matrix)"""
    def __init__(self, lamb: float, mu: float, size: int = 101):
        if lamb < 0 or mu < 0:
            raise ValueError("Parameters cannot be negative.")
        if lamb + mu > 1.0:
            raise ValueError("Total probability lambda + mu cannot exceed 1.0.")

        self._size = size
        self.transition_mat = np.diag([1.0 - lamb - mu] * self._size, k=0)
        self.transition_mat += np.diag([mu] * (self._size - 1), k=-1)
        self.transition_mat += np.diag([lamb] * (self._size - 1), k=1)
        self.transition_mat[0, 0] += mu     # Boundary condition correction
        self.transition_mat[-1, -1] += lamb # Boundary condition correction

    def projectForward(self, n_step: int, p0: np.ndarray) -> np.ndarray:
        return StochasticAlgorithms.dtmc_matrix_project(self.transition_mat, p0, n_step)

    def computeTraj(self, n_step: int, p0: np.ndarray) -> np.ndarray:
        return StochasticAlgorithms.dtmc_matrix_trajectory(self.transition_mat, p0, n_step)


class OSLog:
    """Oversimplified Logistic System (Infinite State DTMC via Trajectory Sampling)"""
    def __init__(self, lamb: float, mu: float, k: int):
        if lamb < 0 or mu < 0 or k < 0:
            raise ValueError("Parameters cannot be negative.")
        if lamb + mu > 1.0:
            raise ValueError("Total probability lambda + mu cannot exceed 1.0.")

        self._lambda = lamb
        self._mu = mu
        self._K = k
        self._coeff = (lamb - mu) / k
        # Normalization constant based on Nmax = 2K
        self._norm = lamb * k * 2 + mu * k * 2 + self._coeff * k * k * 4
        print("EQUIVALENT TIME UNIT: ", 1.0 / self._norm)

    def __computeInitialState(self, p0: np.ndarray) -> int:
        return StochasticAlgorithms.dtmc_sample_state(p0)

    def computeTraj(self, n_step: int, p0: np.ndarray) -> np.ndarray:
        if n_step <= 0:
            raise ValueError("n_step must be positive.")
        if not np.isclose(p0.sum(), 1.0) or np.any(p0 < 0):
            raise ValueError("p0 must be a valid probability distribution.")

        output = np.zeros(n_step + 1, dtype=float)
        output[0] = self.__computeInitialState(p0)

        # Sampling step transition probabilities dynamically
        def transition_probs(state):
            # Normalize rates to probabilities using the time-scaling constant
            l_N = self._lambda * state / self._norm
            m_N = (self._mu * state + self._coeff * state**2) / self._norm
            p_stay = 1.0 - (l_N + m_N)
            return l_N, m_N, p_stay

        # Run trajectory sampler
        p0_dummy = np.zeros(int(output[0]) + 1)
        p0_dummy[int(output[0])] = 1.0
        sampled = StochasticAlgorithms.dtmc_sample_trajectory(n_step, p0_dummy, transition_probs)
        return sampled.astype(float)


class CAsystem:
    """Cellular Automata System (Von Neumann/Rook Neighbourhood)"""
    def __init__(self, r1: float, r2: float):
        self._size = 64
        self._shape = 8
        self._rho1 = r1
        self._rho2 = r2
        self._adjList = self.__createAdjList()

    def __createAdjList(self) -> List[List[int]]:
        adj = []
        for i in range(self._size):
            local_adj = []
            if i // self._shape != 0:                      # Not in first row (Top)
                local_adj.append(i - self._shape)
            if i % self._shape != 0:                       # Not in first column (Left)
                local_adj.append(i - 1)
            if (i + 1) % self._shape != 0:                 # Not in last column (Right)
                local_adj.append(i + 1)
            if i // self._shape != self._shape - 1:        # Not in last row (Bottom)
                local_adj.append(i + self._shape)
            adj.append(local_adj)
        return adj

    def __probF(self, K: int) -> float:
        return self._rho1 * K / 2.0 if K in (1, 2) else 0.0

    def __probG(self, K: int) -> float:
        return self._rho2 * K / 2.0 if K in (3, 4) else 0.0

    def __evolveCA(self, s: int, k: int) -> int:
        r = random.uniform(0, 1)
        if s == 0:
            return 1 if r <= self.__probF(k) else 0
        else:
            return 0 if r <= self.__probG(k) else 1

    def computeTraj(self, n_step: int, s0: np.ndarray) -> np.ndarray:
        if n_step < 0:
            raise ValueError("Steps cannot be negative.")
        if len(s0) != self._size:
            raise ValueError("Initial status vector dimensions must match grid size.")
        if not np.all((s0 == 0) | (s0 == 1)):
            raise ValueError("Automata states must be binary (0 or 1).")

        traj = np.zeros((self._size, n_step + 1), dtype=int)
        traj[:, 0] = s0

        for step in range(1, n_step + 1):
            for cell in range(self._size):
                # FIXED CA INDEX BUG: Corrected indexing of actual neighbour states
                K = 0
                for n_idx in self._adjList[cell]:
                    if traj[n_idx, step - 1] == 1:
                        K += 1
                
                traj[cell, step] = self.__evolveCA(traj[cell, step - 1], K)

        return traj


# =========================================================================
# 2. CONTINUOUS-TIME MARKOV CHAINS (CTMC) MODELS
# =========================================================================

class CTMC:
    """Finite CTMC Solver (Kolmogorov Forward Equation solver)"""
    @staticmethod
    def generator_check(M: np.ndarray) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("Infinitesimal generator must be a 2D square matrix.")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if i != j and M[i, j] < 0:
                    raise ValueError(f"Off-diagonal rate Q[{i},{j}] cannot be negative.")
            if not np.isclose(np.sum(M[i, :]), 0.0, atol=1e-6):
                raise ValueError(f"Row {i} of generator matrix must sum to 0.")

    def __infinitesimal_generator_maker(self) -> np.ndarray:
        Q = np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            row_sum = 0.0
            for j in range(self.n_states):
                if i != j:
                    Q[i, j] = random.random()
                    row_sum += Q[i, j]
            Q[i, i] = -row_sum
        return Q

    def __init__(self, N: int):
        self.n_states = N
        self.inf_generator = self.__infinitesimal_generator_maker()
        self.generator_check(self.inf_generator)

    def Kolmogorov_solve(self, p0: np.ndarray, T: float, M: int) -> np.ndarray:
        return StochasticAlgorithms.ctmc_kolmogorov_solve(self.inf_generator, p0, T, M)


class CTMC_SIM:
    """Finite CTMC Simulator (Gillespie SSA Trajectory Sampler)"""
    def __init__(self, N: int):
        self.n_states = N
        self.inf_generator = self.__infinitesimal_generator_maker()
        CTMC.generator_check(self.inf_generator)

    def __infinitesimal_generator_maker(self) -> np.ndarray:
        Q = np.zeros((self.n_states, self.n_states), dtype=float)
        for i in range(self.n_states):
            row_sum = 0.0
            for j in range(self.n_states):
                if i != j:
                    Q[i, j] = random.random()
                    row_sum += Q[i, j]
            Q[i, i] = -row_sum
        return Q

    def compute_prob_dist(self, p0: np.ndarray, T: float, Nsim: int) -> np.ndarray:
        if T == 0:
            return p0
        if T < 0 or Nsim <= 0:
            raise ValueError("Invalid parameters.")

        bins = np.zeros(self.n_states, dtype=float)
        for _ in range(Nsim):
            end_state = StochasticAlgorithms.gillespie_ssa_finite(self.inf_generator, p0, T)
            bins[end_state] += 1.0

        return bins / Nsim


class MALT_SIM:
    """Continuous-Time Markov Chain Malthusian System Simulator"""
    def __init__(self, b: float, d: float):
        if b <= 0 or d <= 0:
            raise ValueError("Rates must be positive.")
        self._b = b
        self._d = d

    def compute_prob_dist(self, p0: np.ndarray, T: float, Nsim: int) -> np.ndarray:
        if T == 0:
            return p0
        if T < 0 or Nsim <= 0:
            raise ValueError("Invalid parameters.")

        states = np.zeros(Nsim, dtype=int)
        birth_fn = lambda N: self._b * N
        death_fn = lambda N: self._d * N

        for i in range(Nsim):
            states[i] = StochasticAlgorithms.gillespie_ssa_population(birth_fn, death_fn, p0, T)

        s_max = np.max(states)
        bins = np.zeros(s_max + 1, dtype=float)
        for val in states:
            bins[val] += 1.0

        return bins / Nsim


class CRN_SIM:
    """Continuous-Time Chemical Reaction Network Simulator"""
    def __init__(self, k1: float, k2: float, k3: float, mu: float, nu: float):
        if k1 <= 0 or k2 <= 0 or k3 <= 0 or mu <= 0 or nu <= 0:
            raise ValueError("Reaction rates must be positive.")
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._mu = mu
        self._nu = nu
        
        # 5 reactions:
        # 0: null -> A
        # 1: A -> B
        # 2: A + B -> C
        # 3: A -> null (degradation)
        # 4: C -> null (degradation)
        self.stoichiometry = np.array([
            [ 1, -1, -1, -1,  0],  # change in A
            [ 0,  1, -1,  0,  0],  # change in B
            [ 0,  0,  1,  0, -1]   # change in C
        ], dtype=float)

    def __getPropensities(self, s: np.ndarray) -> np.ndarray:
        prop = np.zeros(5, dtype=float)
        tot_pop = np.sum(s)
        if tot_pop <= 0:
            return prop

        # Notebook propensity formulation (normalized by total population size)
        prop[0] = self._k1
        prop[1] = self._k2 * s[0] / tot_pop
        prop[2] = self._k3 * (s[0] / tot_pop) * (s[1] / tot_pop)
        prop[3] = self._mu * s[0] / tot_pop
        prop[4] = self._nu * s[2] / tot_pop
        return prop

    def simulationSSA(self, d0: np.ndarray, T: float) -> np.ndarray:
        # Run Gillespie SSA CRN solver
        # Note: Handled custom stoichiometry updates internally inside algorithms.py,
        # but to keep exact mapping with notebook and fix CRN index bug (reaction 3 degrades A):
        states = []
        state = d0.astype(float).copy()
        curr_time = 0.0
        states.append(state.copy())

        while curr_time < T:
            prop = self.__getPropensities(state)
            a0 = np.sum(prop)
            if a0 < 1e-12:
                break

            tau = np.random.exponential(1.0 / a0)
            curr_time += tau
            if curr_time > T:
                break

            # Choose next reaction
            r = random.random()
            cdf_prop = np.cumsum(prop) / a0
            next_R = 0
            for idx, val in enumerate(cdf_prop):
                if r < val:
                    next_R = idx
                    break

            # Apply state update
            if next_R == 0:
                state[0] += 1
            elif next_R == 1:
                state[0] -= 1
                state[1] += 1
            elif next_R == 2:
                state[0] -= 1
                state[1] -= 1
                state[2] += 1
            elif next_R == 3:
                # FIXED CRN UPDATE BUG: Decrements A (state[0]) instead of B (state[1])
                state[0] -= 1
            else:
                state[2] -= 1

            states.append(state.copy())

        return np.array(states)


# =========================================================================
# 3. STOCHASTIC DIFFERENTIAL EQUATIONS (SDE) MODELS
# =========================================================================

class DetLog:
    """Deterministic Logistic Population Model"""
    def __init__(self, r: float, K: float):
        if r < 0 or K <= 0:
            raise ValueError("r must be non-negative and K must be positive.")
        self._r = r
        self._K = K

    def computeTraj(self, n_step: int, h: float, x0: float) -> np.ndarray:
        drift = lambda x: self._r * x * (1.0 - x / self._K)
        return StochasticAlgorithms.euler_deterministic(x0, drift, n_step * h, h)


class StoLog:
    """Stochastic Logistic Population Model (Euler-Maruyama SDE)"""
    def __init__(self, r: float, a: float):
        if r < 0 or a < 0:
            raise ValueError("System parameters must be non-negative.")
        self._r = r
        self._a = a

    def computeTraj(self, n_step: int, h: float, x0: float) -> np.ndarray:
        drift = lambda x: x * (self._r - x)
        diffusion = lambda x: self._a * x
        return StochasticAlgorithms.euler_maruyama(x0, drift, diffusion, n_step * h, h)


class Harvest:
    """Stochastic Harvest Process (supports Milstein, RK4, Splitting, and Control)"""
    def __init__(self, a: float, k: Union[float, np.ndarray], s: float = 0.0):
        if a < 0 or s < 0:
            raise ValueError("Parameters a and s must be non-negative.")
        self._a = a
        self._s = s
        self.setParam(a, k, s)

    def setParam(self, a: float, k: Union[float, np.ndarray], s: float = 0.0) -> None:
        self._a = a
        self._s = s
        if isinstance(k, np.ndarray):
            if np.any(k < 0):
                raise ValueError("All control k values must be non-negative.")
            self._k = k
        else:
            if k < 0:
                raise ValueError("k must be non-negative.")
            self._k = float(k)

    def __RK4_step(self, t_n: float, y_n: float, h: float) -> float:
        """Helper to advance the deterministic part by one RK4 step"""
        # Determine the current value of k (scalar or piecewise constant vector)
        if isinstance(self._k, np.ndarray):
            # FIXED RK4 CONTROL INDEX BUG: Map to correct dynamic region index without modulo wrap
            step_idx = int(round(t_n / h))
            k_val = self._k[min(step_idx // 10, len(self._k) - 1)]
        else:
            k_val = self._k

        drift_fn = lambda x: self._a * x * (1.0 - x) - k_val * x
        
        k1 = drift_fn(y_n)
        k2 = drift_fn(y_n + 0.5 * h * k1)
        k3 = drift_fn(y_n + 0.5 * h * k2)
        k4 = drift_fn(y_n + h * k3)
        return h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def computeTraj(self, n_step: int, h: float, x0: float) -> np.ndarray:
        """
        Computes a trajectory of the Harvest Process.
        Uses Naive Splitting: RK4 for deterministic, Milstein for stochastic.
        """
        traj = np.zeros(n_step + 1, dtype=float)
        traj[0] = x0

        for i in range(1, n_step + 1):
            # Deterministic drift update
            det_change = self.__RK4_step((i - 1) * h, traj[i - 1], h)
            
            # Stochastic diffusion update (Milstein method)
            eps = random.gauss(0, 1)
            stoch_change = self._s * traj[i - 1] * (1.0 - traj[i - 1]) * np.sqrt(h) * eps
            stoch_change += 0.5 * (self._s * traj[i - 1] * (1.0 - traj[i - 1])) * (self._s * (1.0 - 2.0 * traj[i - 1])) * h * (eps * eps - 1.0)
            
            traj[i] = traj[i - 1] + det_change + stoch_change

        return traj


class Telegraphic:
    """Bounded Dichotomous Markov Noise (Telegraphic Noise) Model"""
    def __init__(self, g: float, o: float, PT: float):
        if g < 0 or o < 0:
            raise ValueError("Parameters must be non-negative.")
        if not (0.0 <= PT <= 1.0):
            raise ValueError("PT must be a valid probability.")
        self._gamma = g
        self._omega = o
        self._PT = PT
        self._state = 1

    def __sort_initial_state(self) -> None:
        self._state = -1 if random.uniform(0, 1) <= self._PT else 1

    def __compute_new_state(self, h: float) -> None:
        r = random.uniform(0, 1)
        if self._state == 1:
            if r <= self._PT * h:
                self._state = -1
        else:
            if r <= (1.0 - self._PT) * h:
                self._state = 1

    def computeTraj(self, n_step: int, h: float, x0: float) -> np.ndarray:
        traj = np.zeros(n_step + 1, dtype=float)
        traj[0] = x0
        self.__sort_initial_state()

        for i in range(1, n_step + 1):
            self.__compute_new_state(h)
            # Drift integrated using RK4 (including the piecewise telegraphic noise)
            drift_fn = lambda x: -self._gamma * x + self._omega * self._state
            
            k1 = drift_fn(traj[i - 1])
            k2 = drift_fn(traj[i - 1] + 0.5 * h * k1)
            k3 = drift_fn(traj[i - 1] + 0.5 * h * k2)
            k4 = drift_fn(traj[i - 1] + h * k3)
            det_change = h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
            traj[i] = traj[i - 1] + det_change

        return traj


# =========================================================================
# 4. EVOLUTIONARY IMITATION GAMES & MEMORY MODELS
# =========================================================================

class EvoGameNM:
    """Evolutionary Imitation Game Model without Memory"""
    def __init__(self, x_e: float):
        if not (0.0 <= x_e <= 1.0):
            raise ValueError("x_e must be between 0 and 1.")
        self._x_e = x_e

    def computeTraj(self, n_step: int, h: float, x0: float) -> np.ndarray:
        drift = lambda t, x: x * (1.0 - x) * (self._x_e - x)
        return StochasticAlgorithms.rk4_deterministic(x0, drift, n_step * h, h)


class EvoGameEFK:
    """Evolutionary Imitation Game Model with Exponentially Fading Kernel Memory"""
    def __init__(self, x_e: float, tau: float):
        if not (0.0 <= x_e <= 1.0) or tau <= 0:
            raise ValueError("x_e must be in [0, 1] and tau must be positive.")
        self._x_e = x_e
        self._tau = tau

    def computeTraj(self, n_step: int, h: float, x0: np.ndarray) -> np.ndarray:
        # System:
        # dx/dt = x*(1-x)*(x_e - M_v)
        # dM_v/dt = (x - M_v)/tau
        
        # State vector y = [x, M_v]
        def system_drift(t, y):
            x, M_v = y[0], y[1]
            dx = x * (1.0 - x) * (self._x_e - M_v)
            dM = (x - M_v) / self._tau
            return np.array([dx, dM])

        # FIXED MEMORY RK4 UPDATE BUG: General multi-D RK4 solver handles variables correctly.
        # Calling rk4_deterministic from StochasticAlgorithms ensures correct updates.
        return StochasticAlgorithms.rk4_deterministic(x0, system_drift, n_step * h, h)


# =========================================================================
# 5. CONTROL THEORY (PARTICLE SWARM OPTIMIZATION)
# =========================================================================

class PSO_Manager:
    """Particle Swarm Optimization (PSO) Manager for Static and Dynamic Control"""
    def __init__(self, N: int, w: float, c1: float, c2: float, dynamic: bool = False, Nwin: int = 12):
        if N <= 0 or w < 0 or c1 < 0 or c2 < 0:
            raise ValueError("Invalid parameters.")
        self._N = N
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self._dynamic = dynamic
        self._Nwin = Nwin if dynamic else 1

        # Particle dimensions
        dim = self._Nwin
        if self._dynamic:
            self.k_values = np.zeros((N, dim), dtype=float)
            for i in range(self._N):
                for j in range(dim):
                    self.k_values[i, j] = random.uniform(0, 0.5)
            self.pbest_values = copy.copy(self.k_values)
            self.v_values = np.zeros((N, dim), dtype=float)
        else:
            self.k_values = np.zeros(N, dtype=float)
            for i in range(self._N):
                self.k_values[i] = random.uniform(0, 0.5)
            self.pbest_values = copy.copy(self.k_values)
            self.v_values = np.zeros(N, dtype=float)

        self.loss_values = -1e30 * np.ones(N, dtype=float)
        self._lambda = 5 if not dynamic else 1
        self._k_star = 0.2
        self._var_counter = 0

    def computeLoss(self, T: np.ndarray, index: int, h: float) -> float:
        if self._dynamic:
            loss = 0.0
            for i in range(self._Nwin):
                loss += self.k_values[index, i] - self._k_star
            loss *= -self._lambda

            I_val = 0.0
            counter = -1
            for i in range(len(T)):
                if i % 10 == 0:
                    counter = min(counter + 1, self._Nwin - 1)
                I_val += self.k_values[index, counter] * T[i] * h
            loss += I_val
            return loss
        else:
            loss = -self._lambda * (self.k_values[index] - self._k_star)
            I_val = np.sum(T) * h * self.k_values[index]
            loss += I_val
            return loss

    def setLossValue(self, L: float, index: int) -> None:
        if L >= self.loss_values[index]:
            self.loss_values[index] = L
            if self._dynamic:
                self.pbest_values[index, :] = self.k_values[index, :].copy()
            else:
                self.pbest_values[index] = self.k_values[index]

    def evolveValues(self) -> None:
        gbest_idx = np.argmax(self.loss_values)
        
        if self._dynamic:
            gbest = self.k_values[gbest_idx, :].copy()
            for i in range(self._N):
                r1 = np.random.uniform(0, 1, self._Nwin)
                r2 = np.random.uniform(0, 1, self._Nwin)
                self.v_values[i, :] = self._w * self.v_values[i, :]
                self.v_values[i, :] += self._c1 * r1 * (self.pbest_values[i, :] - self.k_values[i, :])
                self.v_values[i, :] += self._c2 * r2 * (gbest - self.k_values[i, :])
                self.k_values[i, :] += self.v_values[i, :]
        else:
            gbest = self.k_values[gbest_idx]
            for i in range(self._N):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                self.v_values[i] = self._w * self.v_values[i]
                self.v_values[i] += self._c1 * r1 * (self.pbest_values[i] - self.k_values[i])
                self.v_values[i] += self._c2 * r2 * (gbest - self.k_values[i])
                self.k_values[i] += self.v_values[i]

    def exitCondition(self, index: int) -> bool:
        if index < 40:
            return True

        mean_loss = np.mean(self.loss_values)
        if abs(mean_loss) < 1e-12:
            return False
            
        if np.var(self.loss_values) / mean_loss <= 0.01:
            self._var_counter += 1

        if self._var_counter == 40:
            print("Exited through variance condition")
            return False

        if index == 300:
            print("Exited through maximum iteration")
            return False

        return True
