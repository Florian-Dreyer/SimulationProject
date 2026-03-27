"""
Adaptive-network SIR epidemic simulator.

This module simulates an SIR (Susceptible-Infected-Recovered) epidemic
spreading on a network that evolves over time. The key idea is that
susceptible individuals can "rewire" their connections to avoid infected
neighbors, which couples the disease dynamics with the network topology.

The model proceeds in discrete time steps, each with three phases:
  1. Infection: infected nodes transmit the disease to susceptible neighbors
  2. Recovery: infected nodes recover (and become immune)
  3. Rewiring: susceptible nodes break links with infected neighbors and
     form new connections elsewhere

Reference: Gross et al. (2006), "Epidemic dynamics on an adaptive network",
Physical Review Letters, 96(20), 208701.
"""

import numpy as np
from joblib import Parallel, delayed
from numba import njit


def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    """Run one replicate of the adaptive-network SIR model.

    Parameters
    ----------
    beta : float in [0, 1]
        Transmission probability. At each time step, each S-I edge
        transmits the infection independently with probability beta.
        Higher beta means the disease spreads faster.
    gamma : float in [0, 1]
        Recovery probability. At each time step, each infected node
        recovers independently with probability gamma.
        Higher gamma means shorter infectious period (on average 1/gamma steps).
    rho : float in [0, 1]
        Rewiring probability. At each time step, each S-I edge is
        rewired independently with probability rho. The susceptible
        node drops the link to its infected neighbor and connects to
        a randomly chosen new node instead.
        Higher rho means more active social distancing behavior.
    N : int, default=200
        Number of nodes (individuals) in the network.
    p_edge : float, default=0.05
        Probability of an edge between any two nodes in the initial
        Erdos-Renyi random graph. Expected initial degree is (N-1)*p_edge.
        With N=200 and p_edge=0.05, the expected degree is about 10.
    n_infected0 : int, default=5
        Number of nodes infected at time t=0. These are chosen
        uniformly at random (without replacement) from all N nodes.
    T : int, default=200
        Number of discrete time steps to simulate.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility. If None, a new
        generator is created with an arbitrary seed. Pass
        np.random.default_rng(seed) for reproducible runs.

    Returns
    -------
    infected_fraction : np.ndarray, shape (T+1,)
        Fraction of the population that is infected at each time step,
        from t=0 to t=T. Values are in [0, 1].
    rewire_counts : np.ndarray, shape (T+1,)
        Number of successful rewiring events at each time step.
        Always 0 at t=0 (no rewiring before the simulation starts).
    degree_histogram : np.ndarray, shape (31,)
        Histogram of node degrees at the final time step t=T.
        degree_histogram[d] = number of nodes with degree d, for d=0..29.
        degree_histogram[30] counts all nodes with degree >= 30.
    """
    if rng is None:
        rng = np.random.default_rng()

    # =====================================================================
    # STEP 0: Build the initial contact network as an Erdos-Renyi graph.
    #
    # We represent the network as an adjacency list using Python sets.
    # neighbors[i] is the set of node indices connected to node i.
    # Sets allow O(1) lookups for "is j a neighbor of i?" and efficient
    # add/remove operations, which is important for the rewiring step.
    #
    # For each pair (i, j) with i < j, we add an edge with probability
    # p_edge. This produces an undirected graph (if i is connected to j,
    # then j is also connected to i).
    # =====================================================================
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # =====================================================================
    # Initialize the health state of each node.
    #
    # We encode states as integers:
    #   0 = Susceptible (S): can catch the disease
    #   1 = Infected (I):    currently infectious
    #   2 = Recovered (R):   immune, cannot be infected again
    #
    # At t=0, we pick n_infected0 nodes uniformly at random to be infected.
    # All other nodes start as susceptible.
    # =====================================================================
    state = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    # Arrays to record the summary statistics at each time step
    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    # =================================================================
    # Main simulation loop: iterate over T discrete time steps.
    # Each time step has three phases applied in order:
    #   Phase 1: Infection (S -> I transitions)
    #   Phase 2: Recovery  (I -> R transitions)
    #   Phase 3: Rewiring  (network topology changes)
    # =================================================================
    for t in range(1, T + 1):
        # =============================================================
        # PHASE 1: INFECTION (synchronous update)
        #
        # For every infected node i, look at each of its neighbors j.
        # If j is susceptible (state 0), the infection transmits with
        # probability beta.
        #
        # Important: we use synchronous (parallel) updating. We first
        # collect ALL new infections in a set, then apply them all at
        # once. This prevents "chain infections" within a single step
        # (where a newly infected node immediately infects its own
        # neighbors in the same step).
        # =============================================================
        new_infections = set()
        infected_nodes = np.where(state == 1)[0]

        for i in infected_nodes:
            for j in neighbors[i]:
                if state[j] == 0:  # j is susceptible
                    if rng.random() < beta:
                        new_infections.add(j)

        # Apply all new infections at once (synchronous update)
        for j in new_infections:
            state[j] = 1

        # =============================================================
        # PHASE 2: RECOVERY
        #
        # Each currently infected node (including those just infected
        # in Phase 1) recovers independently with probability gamma.
        # Recovery is permanent: recovered nodes move to state 2 (R)
        # and can never be infected again.
        #
        # We recompute the infected set to include newly infected nodes.
        # =============================================================
        infected_nodes = np.where(state == 1)[0]
        for i in infected_nodes:
            if rng.random() < gamma:
                state[i] = 2

        # =============================================================
        # PHASE 3: NETWORK REWIRING (adaptive behavior)
        #
        # This is what makes the model "adaptive": the network structure
        # changes in response to the disease.
        #
        # We look at all edges between a susceptible node (S) and an
        # infected node (I), called "S-I edges". For each such edge,
        # with probability rho, the susceptible node:
        #   1. Drops the connection to its infected neighbor
        #   2. Forms a new connection to a randomly chosen other node
        #      (that it is not already connected to)
        #
        # This models social distancing: susceptible individuals
        # actively avoid infected contacts.
        # =============================================================
        rewire_count = 0

        # First, collect all S-I edges. We iterate over susceptible
        # nodes and check their neighbors for infected ones.
        si_edges = []
        for i in range(N):
            if state[i] == 0:  # node i is susceptible
                for j in neighbors[i]:
                    if state[j] == 1:  # neighbor j is infected
                        si_edges.append((i, j))

        # Process each S-I edge for potential rewiring
        for s_node, i_node in si_edges:
            if rng.random() < rho:
                # Check that this edge still exists. An earlier rewiring
                # in this same loop may have already removed it (since
                # rewiring can affect shared neighborhoods).
                if i_node not in neighbors[s_node]:
                    continue

                # Remove the S-I edge (break the link in both directions)
                neighbors[s_node].discard(i_node)
                neighbors[i_node].discard(s_node)

                # Find all valid candidates for a new connection:
                # any node that is not s_node itself and not already
                # a neighbor of s_node. Note that the new partner can
                # be in any state (S, I, or R).
                candidates = []
                for k in range(N):
                    if k != s_node and k not in neighbors[s_node]:
                        candidates.append(k)

                # If there is at least one valid candidate, pick one
                # uniformly at random and create the new edge
                if candidates:
                    new_partner = rng.choice(candidates)
                    neighbors[s_node].add(new_partner)
                    neighbors[new_partner].add(s_node)
                    rewire_count += 1

        # Record summary statistics for this time step
        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    # =====================================================================
    # Compute the degree histogram at the final time step.
    #
    # The degree of a node is its number of connections (neighbors).
    # We bin degrees from 0 to 29 individually, and lump all degrees >= 30
    # into a single bin (index 30). This gives a fixed-size output array
    # of shape (31,) regardless of the actual degree distribution.
    # =====================================================================
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = min(len(neighbors[i]), 30)
        degree_histogram[deg] += 1

    return infected_fraction, rewire_counts, degree_histogram


# ── Numba JIT kernel ──────────────────────────────────────────────────────────


@njit(cache=True, nogil=True)
def _simulate_core(beta, gamma, rho, N, p_edge, n_infected0, T, seed):
    """JIT-compiled simulation kernel.

    Differences from the reference implementation
    ---------------------------------------------
    * The adjacency list of Python sets is replaced by a dense boolean
      adjacency matrix (shape N×N, ~40 KB for N=200).  Numba cannot compile
      Python sets, while dense arrays allow tight, cache-friendly loops.
    * np.random.seed(seed) is used instead of np.random.default_rng() because
      Numba's numpy random API does not support the Generator interface.
    * The algorithm (infection / recovery / rewiring rules) is identical to
      the reference.

    Parameters
    ----------
    beta, gamma, rho : float
        Model parameters (see simulate() docstring).
    N : int
        Population size.
    p_edge : float
        Erdos-Renyi edge probability.
    n_infected0 : int
        Number of initially infected nodes.
    T : int
        Number of time steps.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    infected_fraction : np.ndarray, shape (T+1,)
    rewire_counts     : np.ndarray, shape (T+1,)
    degree_histogram  : np.ndarray, shape (31,)
    """
    np.random.seed(seed)

    # ── Build Erdos-Renyi contact network ─────────────────────────────────────
    adj = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                adj[i, j] = True
                adj[j, i] = True

    # ── Initialise health states (0=S, 1=I, 2=R) ─────────────────────────────
    state = np.zeros(N, dtype=np.int8)

    # Pick n_infected0 distinct nodes via a partial Fisher-Yates shuffle
    indices = np.arange(N)
    for k in range(n_infected0):
        swap = k + np.random.randint(0, N - k)
        tmp = indices[k]
        indices[k] = indices[swap]
        indices[swap] = tmp
    for k in range(n_infected0):
        state[indices[k]] = 1

    # ── Output arrays ─────────────────────────────────────────────────────────
    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    # Preallocate work buffers reused every time step (avoids repeated allocs)
    inf_nbrs = np.empty(N, dtype=np.int64)
    candidates = np.empty(N, dtype=np.int64)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(1, T + 1):
        # ── Phase 1: Infection (synchronous update) ───────────────────────────
        # Collect all new infections before applying any state change, so that
        # a node infected in this step cannot immediately infect others.
        new_inf = np.zeros(N, dtype=np.bool_)
        for i in range(N):
            if state[i] == 1:
                for j in range(N):
                    if adj[i, j] and state[j] == 0:
                        if np.random.random() < beta:
                            new_inf[j] = True
        for j in range(N):
            if new_inf[j]:
                state[j] = 1

        # ── Phase 2: Recovery ─────────────────────────────────────────────────
        for i in range(N):
            if state[i] == 1:
                if np.random.random() < gamma:
                    state[i] = 2

        # ── Phase 3: Rewiring (adaptive avoidance) ────────────────────────────
        rewire_count = 0
        for i in range(N):
            if state[i] != 0:  # only susceptible nodes rewire
                continue

            # Snapshot infected neighbours before modifying adj to avoid
            # processing edges that were removed earlier in this step.
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

                # Remove the S-I edge
                adj[i, j] = False
                adj[j, i] = False

                # Collect valid candidates for the new connection
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


# ── Single-run public interface ───────────────────────────────────────────────


def simulate_fast(
    beta: float, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None
):
    """Run one replicate of the adaptive-network SIR model.

    Drop-in replacement for the reference simulator with identical parameters
    and return values.

    Parameters
    ----------
    beta : float in [0, 1]
        Transmission probability per S-I edge per time step.
    gamma : float in [0, 1]
        Recovery probability per infected node per time step.
    rho : float in [0, 1]
        Rewiring probability per S-I edge per time step.
    N : int, default=200
        Population size.
    p_edge : float, default=0.05
        Erdos-Renyi edge probability for the initial graph.
    n_infected0 : int, default=5
        Number of initially infected nodes.
    T : int, default=200
        Number of time steps.
    rng : numpy.random.Generator or None
        Random number generator.  A seed is derived from it when provided;
        a fresh random seed is used otherwise.

    Returns
    -------
    infected_fraction : np.ndarray, shape (T+1,)
        Fraction of the population infected at each time step t=0..T.
    rewire_counts : np.ndarray, shape (T+1,)
        Number of rewiring events at each time step (always 0 at t=0).
    degree_histogram : np.ndarray, shape (31,)
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

'''def simulate_batch(thetas, n_replicates=1, N=200, p_edge=0.05, n_infected0=5,
                   T=200, n_jobs=-1, rng=None):
    """Run many simulations in parallel — the main entry point for ABC.
 
    Parameters
    ----------
    thetas : array-like, shape (M, 3)
        Parameter matrix; each row is (beta, gamma, rho).
    n_replicates : int, default=1
        Number of independent simulation replicates per parameter draw.
        Set to R=40 to match the observed data, which was generated from
        40 replicates per parameter set.  Each replicate uses a different
        random seed so the stochastic variation is independent.
    N, p_edge, n_infected0, T : same as simulate()
        Shared fixed parameters applied to every run.
    n_jobs : int, default=-1
        Number of parallel workers.  -1 uses all available CPU cores.
    rng : numpy.random.Generator or None
        Master RNG used to derive per-run seeds deterministically.
        Pass np.random.default_rng(seed) for fully reproducible batches.
 
    Returns
    -------
    results : list of lists, shape (M, n_replicates)
        results[i][r] is the (infected_fraction, rewire_counts, degree_histogram)
        tuple for parameter draw i and replicate r.
 
    Example
    -------
    >>> rng = np.random.default_rng(42)
    >>> thetas = rng.uniform([0.05, 0.02, 0.0], [0.50, 0.20, 0.8], (1000, 3))
    >>> results = simulate_batch(thetas, n_replicates=40, rng=rng)
    >>> # Access replicate r of parameter draw i:
    >>> infected, rewires, degrees = results[i][r]
    """
    thetas = np.asarray(thetas, dtype=float)
    M = len(thetas)
    if rng is None:
        rng = np.random.default_rng()
 
    seeds = rng.integers(0, 2**31, size=(M, n_replicates))
 
    flat_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_simulate_core)(
            thetas[i, 0], thetas[i, 1], thetas[i, 2],
            int(N), float(p_edge), int(n_infected0), int(T), int(seeds[i, r]),
        )
        for i in range(M)
        for r in range(n_replicates)
    )
 
    # Reshape from (M * n_replicates,) → (M, n_replicates)
    return [
        flat_results[i * n_replicates:(i + 1) * n_replicates]
        for i in range(M)
    ]'''


def simulate_batch(
    thetas, N=200, p_edge=0.05, n_infected0=5, T=200, n_jobs=-1, rng=None
):
    """Run many simulations in parallel — the main entry point for ABC.

    Parameters
    ----------
    thetas : array-like, shape (M, 3)
        Parameter matrix; each row is (beta, gamma, rho).
    N, p_edge, n_infected0, T : same as simulate()
        Shared fixed parameters applied to every run.
    n_jobs : int, default=-1
        Number of parallel workers.  -1 uses all available CPU cores.
    rng : numpy.random.Generator or None
        Master RNG used to derive per-run seeds deterministically.
        Pass np.random.default_rng(seed) for fully reproducible batches.

    Returns
    -------
    list of (infected_fraction, rewire_counts, degree_histogram) tuples,
    one per row of thetas, in the same order.

    Example
    -------
    >>> rng = np.random.default_rng(42)
    >>> prior_samples = rng.uniform([0.05, 0.02, 0.0], [0.50, 0.20, 0.8], (1000, 3))
    >>> results = simulate_batch(prior_samples, rng=rng)
    """
    thetas = np.asarray(thetas, dtype=float)
    M = len(thetas)
    if rng is None:
        rng = np.random.default_rng()

    seeds = rng.integers(0, 2**31, size=M)

    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_simulate_core)(
            thetas[i, 0],
            thetas[i, 1],
            thetas[i, 2],
            int(N),
            float(p_edge),
            int(n_infected0),
            int(T),
            int(seeds[i]),
        )
        for i in range(M)
    )
