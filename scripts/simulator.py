"""
Adaptive-network SIR epidemic simulator.

This module provides three public functions:

    simulate()        — reference pure-Python implementation (slow, readable)
    simulate_fast()   — Numba JIT-compiled single run (~33× faster)
    simulate_batch()  — parallel batch runner for ABC (~387× faster on M3 Max)

The model
---------
A population of N agents interact on an undirected contact network that
evolves over time. Each agent is in one of three states:

    S (Susceptible) — can catch the disease
    I (Infected)    — currently infectious
    R (Recovered)   — permanently immune

The contact network is initialised as an Erdos-Renyi random graph G(N, p_edge).
At each discrete time step, three operations are applied synchronously:

    1. Infection  — each S-I edge transmits the disease with probability β
    2. Recovery   — each infected agent recovers with probability γ
    3. Rewiring   — each S-I edge is broken with probability ρ and the
                    susceptible agent forms a new link to a random non-neighbor
                    (behavioural avoidance / social distancing)

Reference: Gross et al. (2006), "Epidemic dynamics on an adaptive network",
Physical Review Letters, 96(20), 208701.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from numpy.typing import NDArray

# Type aliases
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Output type: (infected_fraction, rewire_counts, degree_histogram)
SimResult: TypeAlias = tuple[FloatArray, IntArray, IntArray]


# ── Reference implementation ──────────────────────────────────────────────────


def simulate(
    beta: float,
    gamma: float,
    rho: float,
    N: int = 200,
    p_edge: float = 0.05,
    n_infected0: int = 5,
    T: int = 200,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run one replicate of the adaptive-network SIR model (pure Python).

    This is the readable reference implementation.  Use simulate_fast() or
    simulate_batch() for performance-critical workloads such as ABC.

    Parameters
    ----------
    beta : float
        Transmission probability per S-I edge per time step. Range: [0, 1].
    gamma : float
        Recovery probability per infected agent per time step. Range: [0, 1].
        Expected infectious period is 1/gamma time steps.
    rho : float
        Rewiring probability per S-I edge per time step. Range: [0, 1].
        Models behavioural avoidance: susceptible agents cut links to infected
        neighbors and reconnect elsewhere.
    N : int, default=200
        Population size (number of nodes in the contact network).
    p_edge : float, default=0.05
        Edge probability for the initial Erdos-Renyi graph G(N, p_edge).
        Expected degree is (N-1) * p_edge ≈ 10 for the default parameters.
    n_infected0 : int, default=5
        Number of agents infected at t=0, chosen uniformly at random.
    T : int, default=200
        Number of discrete time steps to simulate.
    rng : np.random.Generator or None, optional
        Random number generator.  Pass np.random.default_rng(seed) for
        reproducible results.  A fresh generator is used when None.

    Returns
    -------
    infected_fraction : FloatArray, shape (T+1,)
        Fraction of agents in state I at each time step t = 0, ..., T.
    rewire_counts : IntArray, shape (T+1,)
        Number of successful rewiring events at each time step.
        Always 0 at t=0 (rewiring only occurs during the simulation).
    degree_histogram : IntArray, shape (31,)
        Node-degree histogram at t=T.  Bin d counts agents with exactly d
        contacts for d = 0, ..., 29; bin 30 captures all degrees >= 30.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Build initial Erdos-Renyi contact network ─────────────────────────────
    # Represented as an adjacency list of sets for O(1) edge lookups and
    # efficient add/remove during rewiring.
    neighbors: list[set[int]] = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # ── Initialise agent states ───────────────────────────────────────────────
    # 0 = S, 1 = I, 2 = R
    state: NDArray[np.int8] = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    infected_fraction: FloatArray = np.zeros(T + 1)
    rewire_counts: IntArray = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    # ── Main simulation loop ──────────────────────────────────────────────────
    for t in range(1, T + 1):
        # Phase 1: Infection (synchronous — collect all new infections first)
        new_infections: set[int] = set()
        for i in np.where(state == 1)[0]:
            for j in neighbors[i]:
                if state[j] == 0 and rng.random() < beta:
                    new_infections.add(j)
        for j in new_infections:
            state[j] = 1

        # Phase 2: Recovery
        for i in np.where(state == 1)[0]:
            if rng.random() < gamma:
                state[i] = 2

        # Phase 3: Rewiring (adaptive avoidance)
        rewire_count = 0
        si_edges = [
            (i, j)
            for i in range(N)
            if state[i] == 0
            for j in neighbors[i]
            if state[j] == 1
        ]
        for s_node, i_node in si_edges:
            if rng.random() < rho:
                if i_node not in neighbors[s_node]:
                    continue  # edge already removed earlier this step
                neighbors[s_node].discard(i_node)
                neighbors[i_node].discard(s_node)
                candidates = [
                    k for k in range(N) if k != s_node and k not in neighbors[s_node]
                ]
                if candidates:
                    new_partner = rng.choice(candidates)
                    neighbors[s_node].add(new_partner)
                    neighbors[new_partner].add(s_node)
                    rewire_count += 1

        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    # ── Degree histogram at t=T ───────────────────────────────────────────────
    degree_histogram: IntArray = np.zeros(31, dtype=np.int64)
    for i in range(N):
        degree_histogram[min(len(neighbors[i]), 30)] += 1

    return infected_fraction, rewire_counts, degree_histogram


# ── Numba JIT kernel ──────────────────────────────────────────────────────────


@njit(cache=True, nogil=True)
def _simulate_core(
    beta: float,
    gamma: float,
    rho: float,
    N: int,
    p_edge: float,
    n_infected0: int,
    T: int,
    seed: int,
) -> SimResult:
    """JIT-compiled inner simulation kernel (not part of the public API).

    Implements the same algorithm as simulate() with two adaptations required
    by Numba:

    * Adjacency sets -> dense boolean matrix of shape (N, N).  Numba cannot
      compile Python sets; dense arrays allow tight, cache-friendly loops and
      fit in CPU cache (~40 KB for N=200).
    * np.random.default_rng() -> np.random.seed() + np.random.random().
      Numba's NumPy random API does not support the Generator interface.

    Decorated with nogil=True so that the GIL is released during execution,
    enabling true thread parallelism when called from simulate_batch().

    Parameters
    ----------
    beta, gamma, rho : float
        Model parameters — see simulate() for full descriptions.
    N : int
        Population size.
    p_edge : float
        Erdos-Renyi edge probability.
    n_infected0 : int
        Number of initially infected agents.
    T : int
        Number of time steps.
    seed : int
        Random seed.  Derived from a master RNG in simulate_fast() and
        simulate_batch() to ensure reproducibility across parallel runs.

    Returns
    -------
    infected_fraction : FloatArray, shape (T+1,)
    rewire_counts : IntArray, shape (T+1,)
    degree_histogram : IntArray, shape (31,)
    """
    np.random.seed(seed)

    # ── Build Erdos-Renyi contact network ─────────────────────────────────────
    adj = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                adj[i, j] = True
                adj[j, i] = True

    # ── Initialise agent states (0=S, 1=I, 2=R) ──────────────────────────────
    state = np.zeros(N, dtype=np.int8)
    # Partial Fisher-Yates shuffle to select n_infected0 distinct nodes
    # (np.random.choice without replacement is not supported by Numba)
    indices = np.arange(N)
    for k in range(n_infected0):
        swap = k + np.random.randint(0, N - k)
        indices[k], indices[swap] = indices[swap], indices[k]
    for k in range(n_infected0):
        state[indices[k]] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    # Preallocate per-step work buffers to avoid repeated heap allocations
    inf_nbrs = np.empty(N, dtype=np.int64)
    candidates = np.empty(N, dtype=np.int64)

    # ── Main simulation loop ──────────────────────────────────────────────────
    for t in range(1, T + 1):
        # Phase 1: Infection (synchronous)
        new_inf = np.zeros(N, dtype=np.bool_)
        for i in range(N):
            if state[i] == 1:
                for j in range(N):
                    if adj[i, j] and state[j] == 0 and np.random.random() < beta:
                        new_inf[j] = True
        for j in range(N):
            if new_inf[j]:
                state[j] = 1

        # Phase 2: Recovery
        for i in range(N):
            if state[i] == 1 and np.random.random() < gamma:
                state[i] = 2

        # Phase 3: Rewiring
        rewire_count = 0
        for i in range(N):
            if state[i] != 0:
                continue  # only susceptible agents rewire

            # Snapshot infected neighbours before modifying adj so that edges
            # removed earlier in this step are not processed twice.
            n_inf = 0
            for j in range(N):
                if adj[i, j] and state[j] == 1:
                    inf_nbrs[n_inf] = j
                    n_inf += 1

            for idx in range(n_inf):
                j = inf_nbrs[idx]
                if not adj[i, j]:  # already rewired this step
                    continue
                if np.random.random() >= rho:
                    continue

                adj[i, j] = False
                adj[j, i] = False

                n_cand = 0
                for k in range(N):
                    if k != i and not adj[i, k]:
                        candidates[n_cand] = k
                        n_cand += 1

                if n_cand > 0:
                    new_partner = candidates[np.random.randint(0, n_cand)]
                    adj[i, new_partner] = True
                    adj[new_partner, i] = True
                    rewire_count += 1

        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    # ── Degree histogram at t=T ───────────────────────────────────────────────
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = 0
        for j in range(N):
            if adj[i, j]:
                deg += 1
        degree_histogram[min(deg, 30)] += 1

    return infected_fraction, rewire_counts, degree_histogram


# ── Single-run public interface (fast) ───────────────────────────────────────


def simulate_fast(
    beta: float,
    gamma: float,
    rho: float,
    N: int = 200,
    p_edge: float = 0.05,
    n_infected0: int = 5,
    T: int = 200,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run one replicate using the Numba JIT-compiled kernel (~33x faster).

    Drop-in replacement for simulate() with identical parameters and return
    values.  The first call incurs a one-time JIT compilation cost (~10-15 s);
    all subsequent calls within the same session use the cached binary.
    The compiled binary is persisted to disk (cache=True) so the compilation
    cost is only paid once across sessions.

    Parameters
    ----------
    beta : float
        Transmission probability per S-I edge per time step. Range: [0, 1].
    gamma : float
        Recovery probability per infected agent per time step. Range: [0, 1].
    rho : float
        Rewiring probability per S-I edge per time step. Range: [0, 1].
    N : int, default=200
        Population size.
    p_edge : float, default=0.05
        Erdos-Renyi edge probability for the initial graph.
    n_infected0 : int, default=5
        Number of initially infected agents.
    T : int, default=200
        Number of time steps.
    rng : np.random.Generator or None, optional
        Random number generator.  A seed is derived from it when provided;
        a fresh random seed is used otherwise.

    Returns
    -------
    infected_fraction : FloatArray, shape (T+1,)
        Fraction of agents infected at each time step t = 0, ..., T.
    rewire_counts : IntArray, shape (T+1,)
        Number of rewiring events at each time step (0 at t=0).
    degree_histogram : IntArray, shape (31,)
        Node-degree histogram at t=T; bin 30 captures degrees >= 30.
    """
    if rng is None:
        rng = np.random.default_rng()
    seed = int(rng.integers(0, 2**31))
    return _simulate_core(
        float(beta),
        float(gamma),
        float(rho),
        int(N),
        float(p_edge),
        int(n_infected0),
        int(T),
        seed,
    )


# ── Parallel batch interface ──────────────────────────────────────────────────


def simulate_batch(
    thetas: FloatArray,
    n_replicates: int = 1,
    N: int = 200,
    p_edge: float = 0.05,
    n_infected0: int = 5,
    T: int = 200,
    n_jobs: int = -1,
    rng: np.random.Generator | None = None,
) -> list[list[SimResult]]:
    """Run many simulations in parallel — the main entry point for ABC.

    Combines Numba JIT compilation with joblib thread parallelism for a
    ~387x speedup over the pure-Python reference on Apple Silicon (M3 Max).
    nogil=True on _simulate_core releases the GIL during JIT-compiled
    execution, enabling true thread parallelism.

    The batch is dispatched as a single flat list of M * n_replicates tasks
    to keep all threads saturated throughout the run.

    Parameters
    ----------
    thetas : FloatArray, shape (M, 3)
        Parameter matrix; each row is (beta, gamma, rho).
    n_replicates : int, default=1
        Number of independent simulation replicates per parameter draw.
        Set to R=40 to match the observed data (40 replicates per parameter
        set), so that summary statistics are computed over the same number of
        replicates as the observed summaries.  Each replicate uses an
        independent random seed.
    N : int, default=200
        Population size, shared across all runs.
    p_edge : float, default=0.05
        Erdos-Renyi edge probability, shared across all runs.
    n_infected0 : int, default=5
        Number of initially infected agents, shared across all runs.
    T : int, default=200
        Number of time steps, shared across all runs.
    n_jobs : int, default=-1
        Number of parallel worker threads.  -1 uses all available CPU cores.
    rng : np.random.Generator or None, optional
        Master RNG used to derive per-run seeds deterministically.
        Pass np.random.default_rng(seed) for fully reproducible batches.

    Returns
    -------
    results : list[list[SimResult]], shape (M, n_replicates)
        results[i][r] is the SimResult tuple
        (infected_fraction, rewire_counts, degree_histogram)
        for parameter draw i and replicate r, in the same order as thetas.

    Example
    -------
    >>> rng = np.random.default_rng(42)
    >>> thetas = rng.uniform([0.05, 0.02, 0.0], [0.50, 0.20, 0.8], (1000, 3))
    >>> results = simulate_batch(thetas, n_replicates=40, rng=rng)
    >>> infected, rewires, degrees = results[0][0]   # draw 0, replicate 0
    >>> # Stack infected fractions for draw 0 across all 40 replicates:
    >>> infected_fractions = np.array([results[0][r][0] for r in range(40)])
    """
    thetas = np.asarray(thetas, dtype=float)
    M = len(thetas)
    if rng is None:
        rng = np.random.default_rng()

    # Repeat each row of thetas n_replicates times so joblib receives a single
    # flat list of M * n_replicates tasks — this keeps all threads saturated.
    thetas_rep = np.repeat(thetas, n_replicates, axis=0)  # shape (M * R, 3)
    seeds = rng.integers(0, 2**31, size=M * n_replicates)

    flat_results: list[SimResult] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_simulate_core)(
            thetas_rep[k, 0],
            thetas_rep[k, 1],
            thetas_rep[k, 2],
            int(N),
            float(p_edge),
            int(n_infected0),
            int(T),
            int(seeds[k]),
        )
        for k in range(M * n_replicates)
    )

    # Reshape (M * n_replicates,) -> (M, n_replicates)
    return [flat_results[i * n_replicates : (i + 1) * n_replicates] for i in range(M)]
