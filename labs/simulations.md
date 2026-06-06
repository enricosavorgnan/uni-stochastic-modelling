# Stochastic Modelling & Simulation Analysis

This document provides a comprehensive and exhaustive summary of the stochastic processes, simulation algorithms, mathematical formulations, qualitative analyses, and control schedules detailed in the seven laboratory lectures. All relevant formulas have been verified against analytical derivations and implemented in [algorithms.py](file:///D:/university/stochastic-modelling/labs/algorithms.py) and [models.py](file:///D:/university/stochastic-modelling/labs/models.py).

---

## Table of Contents
1. [Lecture 1: Discrete-Time Markov Chains (DTMC)](#lecture-1-discrete-time-markov-chains-dtmc)
2. [Lecture 2: Continuous-Time Markov Chains (CTMC) & Gillespie SSA](#lecture-2-continuous-time-markov-chains-ctmc--gillespie-ssa)
3. [Lecture 3: Stochastic Differential Equations (SDE) & Euler-Maruyama](#lecture-3-stochastic-differential-equations-sde--euler-maruyama)
4. [Lecture 4: Modern SDE Methods & SDE Qualitative Analysis](#lecture-4-modern-sde-methods--sde-qualitative-analysis)
5. [Lecture 5: SDE Errors, Divergences, and telegraphic Noise](#lecture-5-sde-errors-divergences-and-telegraphic-noise)
6. [Lecture 6: Non-Markovian Systems & Memory Kernels](#lecture-6-non-markovian-systems--memory-kernels)
7. [Lecture 7: Control Theory in Dynamical & Stochastic Systems](#lecture-7-control-theory-in-dynamical--stochastic-systems)
8. [Comprehensive Bug Log and Corrections](#comprehensive-bug-log-and-corrections)

---

## Lecture 1: Discrete-Time Markov Chains (DTMC)

Discrete-Time Markov Chains represent stochastic systems that evolve through discrete steps. The transition probabilities depend only on the current state (Markov property).

### 1. Oversimplified Malthusian System
The population size is defined on a finite state space $N(t) \in \{0, 1, \dots, N_{\text{max}}\}$. The dynamics at each time step are governed by growth rate $\lambda$ and death rate $\mu$:

$$
\begin{cases}
N \xrightarrow{p=\lambda} & N+1 \\
N \xrightarrow{p=\mu} & N-1 \\
N \xrightarrow{p=1-\mu-\lambda} & N
\end{cases}
$$

*   **Boundary Conditions:** At the boundaries, the flows are modified. For $N = 0$, $p(0 \to 1) = \lambda$ and $p(0 \to 0) = 1-\lambda$. For $N = N_{\text{max}}$, $p(N_{\text{max}} \to N_{\text{max}}-1) = \mu$ and $p(N_{\text{max}} \to N_{\text{max}}) = 1-\mu$.
*   **Transition Matrix Approach:** If the state space is finite, the system state is represented as a probability vector $\mathbf{p}(t)$. The forward projection is:
    
    $$ \mathbf{p}(t+n) = \mathbf{p}(t) \mathbf{P}^n $$
    
    where $\mathbf{P}$ is the tridiagonal transition matrix.
*   **Qualitative & Steady State Analysis:** The master equation in the interior states is:
    
    $$ \frac{\Delta P(n,t)}{\Delta t} = \lambda P(n-1,t) + \mu P(n+1,t) - (\lambda+\mu)P(n,t) $$
    
    At steady state ($\Delta P/\Delta t = 0$), dividing by $\mu$ yields:
    
    $$ r P_s(n-1) + P_s(n+1) - (1+r)P_s(n) = 0 \quad \text{where} \quad r = \frac{\lambda}{\mu} $$
    
    Rearranging terms:
    
    $$ r(P_s(n-1) - P_s(n)) + (P_s(n+1) - P_s(n)) = 0 $$
    
    *   **Case $r > 1$ ($\lambda > \mu$):** The upper flow dominates. The system accumulates probability towards the upper boundary $N_{\text{max}}$.
    *   **Case $r < 1$ ($\lambda < \mu$):** The lower flow dominates. The population goes extinct, concentrating probability at $N = 0$.
    *   **Case $r = 1$ ($\lambda = \mu$):** The equation simplifies to $P_s(n+1) + P_s(n-1) - 2P_s(n) = 0$. By induction, $P_s(n) = c$ (constant), representing a uniform distribution over the domain. The system behaves as a pure random walk.

### 2. Logistic System
To capture resource limitation without setting an arbitrary boundary $N_{\text{max}}$, we model birth and death rates dependent on population size $N$:

$$
\begin{cases}
N \xrightarrow{p = \lambda_N} & N+1 \\
N \xrightarrow{p = \mu_N} & N-1 \\
N \xrightarrow{p= 1 - \mu_N - \lambda_N} & N
\end{cases}
$$

where $\lambda_N = \lambda N$ and $\mu_N = \mu N + \frac{\lambda - \mu}{K} N^2$.
*   **Dimensionality and Normalization:** Because the rates grow quadratically, they can exceed 1.0, meaning they are rates, not probabilities. To translate this into a DTMC, we must normalize the rates by a scaling factor $\mathcal{M}$ computed at an assumed maximum population size $N_{\text{max}} = 2K$:
    
    $$ \mathcal{M} = \lambda N_{\text{max}} + \mu N_{\text{max}} + \frac{\lambda - \mu}{K} N_{\text{max}}^2 $$
    
    This normalization rescales the time unit ($\delta t = 1/\mathcal{M}$), reducing execution speed but preserving steady-state properties.
*   **Numerical Limitation:** Because transition matrices for infinite state spaces are intractable, we abandon the matrix approach and transition to **trajectory sampling** (Monte Carlo simulation) to reconstruct the probability density function (PDF).

### 3. Cellular Automata (CA)
Cellular automata are spatially discrete dynamical systems. We model 64 binary automata in an $8 \times 8$ grid under a Von Neumann (Rook) neighborhood:

$$ \text{Neigh}(x,y) = \{(x-1,y), (x+1,y), (x,y-1), (x,y+1)\} $$

The state transitions are state-dependent and count-dependent (where $K$ is the number of active neighbors):
*   **Deactivated to Activated ($0 \to 1$):** $P_1(K) = \frac{\rho_1}{2} K$ if $K \in \{1, 2\}$, else $0$.
*   **Activated to Deactivated ($1 \to 0$):** $P_0(K) = \frac{\rho_2}{2} (K-2)$ if $K \in \{3, 4\}$, else $0$.

*Note: The original notebook code contained an indexing bug that checked sequential loop indices instead of the actual neighboring cell states. This has been corrected in [models.py](file:///D:/university/stochastic-modelling/labs/models.py).*

---

## Lecture 2: Continuous-Time Markov Chains (CTMC) & Gillespie SSA

Continuous-Time Markov Chains model transitions occurring at continuously distributed random times.

### 1. Finite System and Forward Kolmogorov Equation
For a finite CTMC of $S$ states, the probability vector $\mathbf{P}(t)$ evolves according to the generator matrix $\mathbf{Q}$ (where row sums are 0, and off-diagonal elements are positive transition rates):

$$ \frac{d\mathbf{P}(t)}{dt} = \mathbf{P}(t)\mathbf{Q} \quad \implies \quad \mathbf{P}(t) = \mathbf{P}(0) \exp(\mathbf{Q}t) $$

*   **Steady State:** Systems with fully connected networks (no absorbent states) converge rapidly to an invariant measure $\boldsymbol{\pi}\mathbf{Q} = 0$.

### 2. Trajectory Sampling: Gillespie SSA
For systems with large or infinite state spaces, the **Gillespie Stochastic Simulation Algorithm (SSA)** yields exact trajectories:
1.  **Jump Chain Construction:** Normalizing the generator matrix $\mathbf{Q}$ yields the jump transition matrix $\mathbf{J}$:
    
    $$ J_{ij} = \frac{Q_{ij}}{-Q_{ii}} \quad \text{for } i \neq j, \quad J_{ii} = 0 $$
    
2.  **State Transition:** Randomly select the next state index using the probabilities in the row $\mathbf{J}[S_{\text{curr}}, :]$.
3.  **Time Step Selection:** The dwell time in the current state is exponentially distributed with rate $q_{\text{tot}} = -Q_{ii}$. It is sampled via inversion of the exponential cumulative distribution function:
    
    $$ \delta t = -\frac{\ln(1-r)}{q_{\text{tot}}} \quad \text{where} \quad r \sim \text{Unif}(0, 1) $$

### 3. Chemical Reaction Networks (CRN)
CRNs model the interactions of molecular populations. In continuous time, propensities $a_j(s)$ govern the reaction rates:
*   $\emptyset \xrightarrow{k_1} A$: propensity $a_0 = k_1$
*   $A \xrightarrow{k_2} B$: propensity $a_1 = k_2 \frac{N_A}{N_{\text{tot}}}$
*   $A + B \xrightarrow{k_3} C$: propensity $a_2 = k_3 \frac{N_A N_B}{N_{\text{tot}}^2}$
*   $A \xrightarrow{\mu} \emptyset$: propensity $a_3 = \mu \frac{N_A}{N_{\text{tot}}}$
*   $C \xrightarrow{\nu} \emptyset$: propensity $a_4 = \nu \frac{N_C}{N_{\text{tot}}}$

*Note: The original notebook code contained an index mismatch for reaction 3, which degraded species B instead of species A. This is fixed in [models.py](file:///D:/university/stochastic-modelling/labs/models.py).*

---

## Lecture 3: Stochastic Differential Equations (SDE) & Euler-Maruyama

Stochastic Differential Equations model continuous variables subject to both deterministic forces and random fluctuations.

### 1. SDE Formulation: The Stochastic Logistic Model
Adding demographic noise to the growth rate of the deterministic logistic ODE $\frac{dx}{dt} = rx(1 - x/K)$ yields the SDE in Ito form:

$$ dx = (rx - x^2)dt + axdW $$

where $dW$ represents the increment of a Wiener process.
*   **Qualitative Analysis:**
    *   If $r - a^2 \geq 0$, the state $x = 0$ is a repulsor, and $x^* = r - a^2$ is the noise-corrected attractor.
    *   If $r - a^2 < 0$, $x = 0$ becomes the attractor for the entire domain. The population undergoes **Noise-Induced Extinction (NIE)** via a transcritical bifurcation.
*   **Stationary Probability Distribution (FPE):** The Fokker-Planck equation yields the analytical stationary distribution $P_s(x)$:
    
    $$ P_s(x) = A x^{\frac{2r}{a^2} - 2} e^{-\frac{2x}{a^2}} $$
    
    *   For $2r/a^2 > 1$, $P_s(0) = 0$ and the distribution is integrable, allowing normalization.
    *   For $2r/a^2 \leq 1$, the distribution diverges at $x = 0$.

### 2. Numerical Integration: Euler-Maruyama
The Euler-Maruyama scheme discretizes SDEs:

$$ x_{n+1} = x_n + h f(x_n) + g(x_n) \sqrt{h} \mathcal{N}(0,1) $$

*   **Error Order:** It has a weak convergence of order $O(h)$ but a strong convergence of order $O(\sqrt{h})$. It requires small steps $h$ to maintain stability near boundaries.
*   **Normalization Constant Integration:** To normalize the stationary PDF, we compute the constant $A = 1/I$ using the trapezoidal rule:
    
    $$ I \approx \sum_{k=0}^{K-1} \frac{P_s(x_{k+1}) + P_s(x_k)}{2} \Delta x_k $$

---

## Lecture 4: Modern SDE Methods & SDE Qualitative Analysis

For systems with highly nonlinear drift or diffusion, we implement higher-order methods and advanced qualitative analyses.

### 1. Stochastic Harvest Process (SHP)
Human harvesting intensity $k$ is added to the logistic equation, with noise perturbing the growth rate $a$:

$$ dx = \left(a x (1-x) - kx\right) dt + s x (1-x) dW $$

### 2. Flow and Bifurcation Analysis
Evaluating the inequality $f(x) \geq g(x) g'(x)$ determines the direction of the drift flow:

$$ a(1-x) - k \geq s^2(1-x)(1-2x) $$

Analyzing boundaries:
*   **At $x \to 0^+$:** Flow is positive if $s \leq \sqrt{a-k}$. If $s > \sqrt{a-k}$, $0$ becomes an attractor.
*   **At $x^* = 1 - k/a$:** For $s > 0$:
    *   If $k/a < 1/2$, the attractor moves right towards $x = 1$.
    *   If $k/a > 1/2$, the attractor moves left towards $0$.
*   **Bifurcation Regimes:**
    *   **$k/a < 1/2$:** Stable attractor moves towards $1$. $0$ becomes an attractor for $s > \sqrt{a-k}$ via a fork bifurcation, creating a repulsor in the interior.
    *   **$k/a > 2/3$:** Attractor moves towards $0$. A transcritical bifurcation swaps stability with $0$, making $0$ the sole attractor.
    *   **$1/2 < k/a < 2/3$:** The fork bifurcation occurs before the transcritical bifurcation. The newly produced repulsor collides with the upper attractor, leading to a ghost bifurcation.

### 3. Higher-Order Solver Schemes
To reduce errors in trajectory simulation:
*   **Milstein Scheme:** Restores the next term of the stochastic Taylor expansion (order $O(h)$):
    
    $$ x_{n+1} = x_n + f(x_n)h + g(x_n)\epsilon\sqrt{h} + \frac{1}{2}g(x_n)g'(x_n)h(\epsilon^2-1) \quad \text{where} \quad \epsilon \sim \mathcal{N}(0,1) $$
    
*   **Naive Splitting:** Integrates the deterministic drift using 4th-order Runge-Kutta (RK4, order $O(h^4)$) and the stochastic diffusion using Milstein:
    
    $$ x_{n+1} = x_n + \text{RK4}_{\text{step}}(x_n, h) + \text{Milstein}_{\text{stochastic}}(x_n, h) $$

---

## Lecture 5: SDE Errors, Divergences, and Telegraphic Noise

Simulating SDEs introduces both numerical cumulative errors and structural meta-errors.

### 1. Analytical Solution of the SHP Fokker-Planck Equation
The stationary PDF is:

$$ P_s(x) = A (1-x)^{-\frac{2(a+s^2-k)}{s^2}} x^{-\frac{2(-a+s^2+k)}{s^2}} e^{\frac{2k}{s^2(x-1)}} $$

### 2. Validation: Kullback-Leibler (KL) Divergence
To measure the error between a simulated distribution $P(x)$ and the analytical distribution $Q(x)$:

$$ D_{KL}(P || Q) = \sum_i P(i) \log_2 \left( \frac{P(i)}{Q(i)} \right) $$

*   **Handling Zero Values:** Since $Q(i) > 0$ is required, we skip elements where the analytical function drops below float precision, which is justified as the simulator will also generate zero samples in these unfeasible regions.

### 3. Telegraphic Noise & Parameter Linking
Telegraphic noise represents a bounded dichotomous Markov process $\eta(t) \in \{1, -1\}$ with transition rate $P_T$.

$$ \frac{dx}{dt} = -\gamma x + \omega \eta(t) $$

*   **Errors:** Integrating the telegraphic noise as a purely stochastic step leads to cumulative shifting of the peaks away from their theoretical positions at $\pm \omega/\gamma$. This error is corrected by embedding the piecewise-constant state directly inside the deterministic RK4 loop.
*   **Total Variation Distance (TVD):** To compare two distributions without the non-zero restriction of KL:
    
    $$ \delta(P, Q) = \frac{1}{2} \sum_{i} |P(i) - Q(i)| $$
    
*   **Parameter Link Analysis:** Plotting the TVD matrix over the parameter grid $(a, k)$ reveals that the system dynamics are invariant to the ratio $k/a$, allowing a reduction in parameter dimensionality.

---

## Lecture 6: Non-Markovian Systems & Memory Kernels

Systems with memory depend on their past trajectory history, violating the Markov property.

### 1. Memory Kernels and Exponentially Fading Kernel (EFK)
We consider the evolutionary imitation game model:

$$ x'(t) = x(1-x)(x_e - M_v(t)) $$

where the history is weighted by a memory kernel:

$$ M_v(t) = \int_0^{\infty} W_v(\tau) x(t-\tau) d\tau $$

Using the Exponentially Fading Kernel $W_v(s) = \frac{1}{\tau} \exp(-s/\tau)$, where $\tau$ is the characteristic memory time, we can transform this non-Markovian system into a 2D system of Markovian ODEs:

$$
\begin{cases}
x' = x(1-x)(x_e - M_v) \\
M_v' = \frac{x - M_v}{\tau}
\end{cases}
$$

*   **Transient Dynamics:** Without memory, the state converges monotonically to $x_e$. With memory, the system exhibits oscillations and damped waves during the transient phase before settling at the steady state.

---

## Lecture 7: Control Theory in Stochastic Systems

Control theory designs input schedules to drive system variables to target states.

### 1. Optimal Control Schedules
*   **Open Loop Control:** Rule-based feedback checking (e.g., threshold triggers).
*   **Static Optimal Control:** Identifying a single optimal value for a parameter (e.g., $k$) and keeping it constant.
*   **Dynamic Optimal Control:** Optimization of a time-varying vector of parameters (e.g., $k(t)$ piecewise constant).

### 2. Loss Function and Constraints
We define the objective function $J$ to maximize woodcutting yield while maintaining the forest biomass near $k^*$:

$$ J = \int_0^T k(t) x(t) dt - \lambda \sum_{i=1}^{N_{\text{win}}} (k_i - k^*) $$

### 3. Swarm Optimization (PSO)
Particle Swarm Optimization uses a population of candidate parameter vectors (particles) that evolve based on their historical best positions ($pbest$) and the global best position ($gbest$):

$$ v_i^{t+1} = w v_i^t + c_1 r_1 (pbest_i - x_i^t) + c_2 r_2 (gbest - x_i^t) $$

*   **Stochastic Control:** If the underlying system is stochastic, we evaluate the loss over $N_{\text{rep}}$ simulated trajectories and optimize based on the mean loss.
*   **Bang-Bang Control:** In dynamic optimal control, the optimizer discovers a high-frequency oscillatory pattern (alternating harvesting intensity) to maximize yield.

---

## Comprehensive Bug Log and Corrections

Several critical bugs and typos present in the laboratory notebooks were identified and fixed in the code files:

### 1. Cellular Automata Neighbor Evaluation Indexing
*   **Location:** `CAsystem.computeTraj` ([1-dtmc.ipynb])
*   **Symptom:** The simulator checked the loop index `k` rather than the neighbor state index `self._adjList[j][k]`, causing the automaton to ignore its actual spatial neighbors.
*   **Correction in models.py:**
    ```python
    # Bug fixed: n_idx fetches the correct neighbor index from self._adjList[cell]
    for n_idx in self._adjList[cell]:
        if traj[n_idx, step - 1] == 1:
            K += 1
    ```

### 2. Chemical Reaction Network SSA Update Mismatch
*   **Location:** `CRN_SIM.simulationSSA` ([2-ctmc.ipynb])
*   **Symptom:** When reaction 3 ($A \xrightarrow{\mu} \emptyset$) was selected, the code decremented species B (`state[1] -= 1`) instead of species A (`state[0] -= 1`).
*   **Correction in models.py:**
    ```python
    elif next_R == 3:
        # Bug fixed: Decrement species A (index 0) instead of B (index 1)
        state[0] -= 1
    ```

### 3. Memory Kernel RK4 Integration Variable Copying
*   **Location:** `EvoGameEFK.__RK4` ([6-errors.ipynb])
*   **Symptom:** During the computation of the coefficient $k_3$, the code passed `y_n[0]` twice, neglecting the memory state variable `y_n[1]`.
*   **Correction in models.py & algorithms.py:** A generic system RK4 solver was implemented in `algorithms.py` which correctly updates all dimensions vectorially, eliminating manual index copying errors:
    ```python
    k_3x = f_1(y_n[0] + h * k_2x / 2.0, y_n[1] + h * k_2y / 2.0)
    k_3y = f_2(y_n[0] + h * k_2x / 2.0, y_n[1] + h * k_2y / 2.0)
    ```

### 4. Dynamic Control Time-Window Mapping
*   **Location:** `Harvest.__RK4` ([7-control-theory.ipynb])
*   **Symptom:** The index of the piecewise-constant control parameters vector `_k` was mapped using `int((t_n/h)%10)`. This modulo operator restricted the index to $\{0, \dots, 9\}$ and looped back, ignoring the final control windows in the 12-dimensional vector.
*   **Correction in models.py:**
    ```python
    # Bug fixed: Map the current step to the correct region index 0 to 11
    step_idx = int(round(t_n / h))
    k_val = self._k[min(step_idx // 10, len(self._k) - 1)]
    ```
