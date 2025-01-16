"""
Module containing the code to infer partial rankings from pairwise interactions
"""

from collections import defaultdict
from copy import deepcopy
from sys import stderr
import time

import numpy as np
import ray
from ray.util.multiprocessing import Pool
from scipy.special import loggamma

# # Uncomment to instantiate ray Pool
# ray.shutdown()
# ray.init(log_to_driver=False)
# pool = Pool()


class DefaultDict(dict):
    """
    default dict that does not add new key when querying a key that does not exist
    """

    def __init__(self, default_factory, **kwargs):
        super().__init__(**kwargs)

        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_factory()
        
def logNcK(n, K):
    """
    Compute the log of the binomial coefficient N choose K
    """
    return loggamma(n + 1) - loggamma(n - K + 1) - loggamma(K + 1)

def partial_rankings(
    N: int,
    M: int,
    e_out: defaultdict,
    e_in: defaultdict,
    TARGET=1e-6,
    force_merge=True,
    exact=True,
    sync=False,
    full_trace=False,
    verbose=False,
):
    """
    Infer partial rankings from pairwise interactions

    Parameters
    ----------
    N : int
        Number of nodes.
    M : int
        Number of edges.
    e_out : defaultdict
        Dictionary of out-edges.
    e_in : defaultdict
        Dictionary of in-edges.
    TARGET : float
        Convergence criterion.
    force_merge : bool
        Force merge of clusters with positive delta DL (if false will stop as soon
        as no merge decreases the DL).
    exact : bool
        Use exact computation of player strengths (sigmas) after every merge.
        If false, will approximate the new strength of the merged cluster as
        the average of the strengths of the two clusters being merged.
    sync : bool
        Use synchronous computation of delta DLs.
    full_trace : bool
        Return a set of results at each merge step. If False, will only return a
        set of results at the DL minimum.
    verbose : bool
        Print verbose output.

    Returns
    -------
    if full_trace:
        trace_list : list
            List of dictionaries containing results at each merge step.
    else:
        results_dict : dict
            Dictionary containing results at the DL minimum.

    Notes
    -----
    As the function contains continuous-valued parameters, all mentions of
    "Description Length" (DL) are to be interpreted as references to the
    negative log-posterior of the model.
    """

    if sync:
        # Initialise pool
        pool = Pool()

    R = N  # Initialise number of unique ranks to the total number of nodes
    # sigmas = np.ones(N)  # Initialise unique rankings

    clusters, n_c, sigmas = (
        {},
        {},
        {},
    )  # dictionaries for clusters, their sizes, and the strengths

    for k in set(e_out.keys()).union(set(e_in.keys())):  # initialise sigmas
        sigmas[k] = 1
        n_c[k] = 1
        clusters[k] = set([k])

    def update_sigmas_bt(
        sigmas: list, out_neigs: defaultdict, in_neigs: defaultdict, TARGET=TARGET
    ):
        """
        Update sigmas via Bradley-Terry self-consistent equations.

        Parameters
        ----------
        sigmas : list
            List of strengths of each cluster.
        out_neigs : defaultdict
            Dictionary of out-neighbours of each cluster.
        in_neigs : defaultdict
            Dictionary of in-neighbours of each cluster.
        TARGET : float
            Convergence criterion.

        Returns
        -------
        float
            New strength of merged cluster
        """

        # Construct all neighbours
        all_neigs = set(out_neigs.keys()).union(set(in_neigs.keys()))

        # Define array of deltas to check for convergence
        deltas = np.ones(len(all_neigs))
        i = 0
        # s_r = 1
        while np.abs(np.max(deltas)) > TARGET:
            i += 1
            for j, r in enumerate(all_neigs):
                # Initialise s_r to 1 if r not in sigmas
                # 1 + sum_s w_{rs} sigma_s / (sigma_r + sigma_s)
                num = 1
                # 2 / (sigma_r + 1) + sum_s w_{sr} / (sigma_r + sigma_s)
                denom = 2 / (sigmas[r] + 1)

                # # Uncomment to force Eq. 27 in newman2023efficient
                # # 1 / (sigma_r + 1) + sum_s w_{rs} sigma_s / (sigma_r + sigma_s)
                # num = 1 / (sigmas[r] + 1)
                # # 1 / (sigma_r + 1) + sum_s w_{sr} / (sigma_r + sigma_s)
                # denom = 1 / (sigmas[r] + 1)

                for s in out_neigs[r].keys():
                    num += (out_neigs[r][s] * sigmas[s]) / (sigmas[r] + sigmas[s])
                for s in in_neigs[r].keys():
                    denom += in_neigs[r][s] / (sigmas[r] + sigmas[s])

                new_sigma = num / denom

                # # Max's convergence criterion
                # # Compute \Delta \sigma_r / \sigma_r
                # delta = (new_sigma - sigmas[r]) / sigmas[r]

                # Mark's convergence criterion
                news = new_sigma / (new_sigma + 1)
                olds = sigmas[r] / (sigmas[r] + 1)
                delta = news - olds

                # Update sigmas[r]
                sigmas[r] = new_sigma

                # Update deltas
                deltas[j] = np.abs(delta)

        if exact:
            return new_sigma

    def get_new_sigma_approx(sigmas, r, s):
        """
        Compute the new strength of the merged cluster (r, s) so as to preserve
        the average win probability between the ranks being merged and the average
        player of strength 1

        Parameters
        ----------
        sigmas : dict
            Dictionary of strengths of each cluster.
        r : int
            Label of first cluster.
        s : int
            Label of second cluster.

        Returns
        -------
        float
            New strength of merged cluster.
        """
        # Compute new sigma
        s_r = sigmas[r]
        s_s = sigmas[s]
        new_sigma_num = (s_r / (s_r + 1)) + (s_s / (s_s + 1))
        new_sigma_denom = 2 - new_sigma_num
        new_sigma = new_sigma_num / new_sigma_denom

        return new_sigma

    # Function definitions for C(R), g(r), and f(r,s)
    def C(R):
        """
        Compute global contribution to the description length

        Parameters
        ----------
        R : int
            Number of unique ranks.

        Returns
        -------
        float
            Glonal contribution to the description length.
        """
        return np.log(N) + logNcK(N - 1, R - 1) + loggamma(N + 1)  # Full prior
        # return logNcK(N - 1, R - 1) + loggamma(N + 1)  # Hard regularization
        # return np.log(N) + logNcK(N - 1, R - 1)  # Soft (network permutation) regularization
        # return logNcK(N - 1, R - 1)  # Prior ignoring constant terms

    def g(r, sigma):
        """
        Compute the node-level contribution to the description length.

        Parameters
        ----------
        r : int
            Label of cluster.
        sigma : float
            Strength of cluster.

        Returns
        -------
        float
            Node-level contribution to the description length.
        """
        if isinstance(r, tuple):
            n_r = n_c[r[0]] + n_c[r[1]]
        else:
            n_r = n_c[r]

        return np.log((sigma + 1) ** 2 / sigma) - loggamma(n_r + 1)  # Logistic prior
        # return ((np.log(sigma)) ** 2) - loggamma(n_r + 1)  # Gaussian prior
        # return np.log((sigma + 1) ** 2 / sigma)  # Network permutation prior

    def f(r, s, sigma_r, sigma_s):
        """
        Compute the interaction contribution to the description length.

        Parameters
        ----------
        r : int
            Label of first cluster.
        s : int
            Label of second cluster.
        sigma_r : float
            Strength of first cluster.
        sigma_s : float
            Strength of second cluster.

        Returns
        -------
        float
            Interaction contribution to the description length.
        """
        if isinstance(r, tuple) and isinstance(s, tuple):
            try:
                e_r0s0 = e_out.get(r[0], 0).get(s[0], 0)
            except AttributeError:
                e_r0s0 = 0
            try:
                e_r0s1 = e_out.get(r[0], 0).get(s[1], 0)
            except AttributeError:
                e_r0s1 = 0
            try:
                e_r1s0 = e_out.get(r[1], 0).get(s[0], 0)
            except AttributeError:
                e_r1s0 = 0
            try:
                e_r1s1 = e_out.get(r[1], 0).get(s[1], 0)
            except AttributeError:
                e_r1s1 = 0
            w_rs = e_r0s0 + e_r0s1 + e_r1s0 + e_r1s1
        elif isinstance(r, tuple):
            w_rs = e_out[r[0]].get(s, 0) + e_out[r[1]].get(s, 0)
        elif isinstance(s, tuple):
            w_rs = e_out[r].get(s[0], 0) + e_out[r].get(s[1], 0)
        else:
            w_rs = e_out[r].get(s, 0)

        return w_rs * np.log((sigma_r + sigma_s) / sigma_r)

    def total_dl():
        """
        Compute the total description length of the model.

        Returns
        -------
        dl : float
            Total description length.
        """
        dl = C(R) + np.sum([g(r, sigmas[r]) for r in n_c.keys()])
        for r in n_c.keys():
            for s in n_c.keys():
                sigma_r = sigmas[r]
                sigma_s = sigmas[s]
                dl += f(r, s, sigma_r, sigma_s)

        return dl

    def delta_dl(r, s, exact=exact):
        """
        Compute the change in the description length of the model when merging clusters r and s

        Parameters
        ----------
        r : int
            Label of first cluster
        s : int
            Label of second cluster

        Returns
        -------
        ddl : float
            Change in description length
        sigma_rs : float
            New strength of merged cluster
        """

        # Check if (r, s) has already been checked
        if not exact:
            if r in ddl_dict:
                if s in ddl_dict[r]:
                    return ddl_dict[r][s]

        # Get in and out neighbours of r and s
        rs_in_neigs = set(e_in[r].keys()).union(set(e_in[s].keys())) - set([r, s])
        rs_out_neigs = set(e_out[r].keys()).union(set(e_out[s].keys())) - set([r, s])
        all_rs_neigs = rs_in_neigs.union(rs_out_neigs)

        # Compute new sigmas for (r, s) merge
        if exact:
            # Update in and out-edges
            new_e_out = defaultdict(dict)
            new_e_out[(r, s)] = defaultdict(dict)
            for t in all_rs_neigs:
                new_e_out[(r, s)][t] = e_out[r].get(t, 0) + e_out[s].get(t, 0)

            new_e_in = defaultdict(dict)
            new_e_in[(r, s)] = defaultdict(dict)
            for t in all_rs_neigs:
                new_e_in[(r, s)][t] = e_in[r].get(t, 0) + e_in[s].get(t, 0)

            new_e_out[(r, s)][(r, s)] = (
                e_out[r].get(r, 0) + e_out[r].get(s, 0) + e_out[s].get(r, 0) + e_out[s].get(s, 0)
            )
            new_e_in[(r, s)][(r, s)] = (
                e_in[r].get(r, 0) + e_in[r].get(s, 0) + e_in[s].get(r, 0) + e_in[s].get(s, 0)
            )

            # Append (r, s): 1 to sigmas dictionary
            sigmas[(r, s)] = 1

            # Compute sigme for merged pair
            rs_sigma = update_sigmas_bt(sigmas, new_e_out, new_e_in)

            # Remove (r, s) from sigma dictionary
            sigmas.pop((r, s))
        else:
            rs_sigma = get_new_sigma_approx(sigmas, r, s)

        # Store sigma in dictionary
        sigma_dict[r] = sigmas[r]
        sigma_dict[s] = sigmas[s]
        sigma_dict[(r, s)] = rs_sigma
        sigma_dict[(s, r)] = rs_sigma

        # Compute delta g
        dg = g((r, s), rs_sigma) - g(r, sigmas[r]) - g(s, sigmas[s])

        # change from flows r to s
        df_internal = (
            f((r, s), (r, s), rs_sigma, rs_sigma)
            - f(r, s, sigmas[r], sigmas[s])
            - f(s, r, sigmas[s], sigmas[r])
            - f(r, r, sigmas[r], sigmas[r])
            - f(s, s, sigmas[s], sigmas[s])
        )

        df_external = 0
        for t in rs_out_neigs:
            df_external += (
                f((r, s), t, rs_sigma, sigmas[t])
                - f(r, t, sigmas[r], sigmas[t])
                - f(s, t, sigmas[s], sigmas[t])
            )
        for t in rs_in_neigs:
            df_external += (
                f(t, (r, s), sigmas[t], rs_sigma)
                - f(t, r, sigmas[t], sigmas[r])
                - f(t, s, sigmas[t], sigmas[s])
            )

        ddl = dg + df_internal + df_external

        # Store delta DL in dictionary
        if not exact:
            if not (r in ddl_dict):
                ddl_dict[r] = {}
            if not (s in ddl_dict):
                ddl_dict[s] = {}
            ddl_dict[r][s] = ddl
            ddl_dict[s][r] = ddl

        # Return in and out neighbours so as not to have to compute them during merge
        return ddl, rs_in_neigs, rs_out_neigs

    def worker(pair):
        """
        Worker function to compute change in description length in parallel

        Parameters
        ----------
        pair : Tuple
            Pair of clusters to merge

        Returns
        -------
        float
            Change in description length
        """
        return delta_dl(pair[0], pair[1], exact=exact)

    def merge_ranks(pair, e_in, e_out, rs_in_neigs, rs_out_neigs, exact=exact):
        """
        Merge clusters r and s into a new cluster rs

        Parameters
        ----------
        pair : tuple
            Tuple of cluster labels to merge

        e_in : defaultdict
            Dictionary of in-edges

        e_out : defaultdict
            Dictionary of out-edges

        rs_in_neigs : set
            Set of in-neighbours of (r, s)

        rs_out_neigs : set
            Set of out-neighbours of (r, s)

        Returns
        -------
        None
        """
        r, s = pair
        rs = str(np.random.randint(100000000))  # new cluster key

        # Update clusters
        clusters[rs] = clusters[r].union(clusters[s])

        # Update cluster sizes
        n_c[rs] = n_c[r] + n_c[s]

        # Compute in and out-neighbours
        all_rs_neigs = rs_in_neigs | rs_out_neigs

        # Initialize once
        e_out_rs = defaultdict(dict)
        e_in_rs = defaultdict(dict)

        # Combine loops and minimize operations
        for t in all_rs_neigs:
            e_out_rs_t = e_out[r].get(t, 0) + e_out[s].get(t, 0)
            e_out_rs[t] = e_out_rs_t
            e_out[t][rs] = e_out[t].get(r, 0) + e_out[t].get(s, 0)

            e_in_rs_t = e_in[r].get(t, 0) + e_in[s].get(t, 0)
            e_in_rs[t] = e_in_rs_t
            e_in[t][rs] = e_in[t].get(r, 0) + e_in[t].get(s, 0)

        # Update dictionaries after loop to minimize operations
        e_out[rs] = e_out_rs
        e_in[rs] = e_in_rs

        # Directly compute self-references
        self_ref = sum(
            [
                e_out[r].get(r, 0),
                e_out[r].get(s, 0),
                e_out[s].get(r, 0),
                e_out[s].get(s, 0),
            ]
        )
        e_out[rs][rs] = e_in[rs][rs] = self_ref

        # Pop references to r and s in in and out-edges
        for t in all_rs_neigs:
            e_out[t].pop(r, None)
            e_out[t].pop(s, None)
            e_in[t].pop(r, None)
            e_in[t].pop(s, None)

        # Remove rest of obsolete terms
        del clusters[r], clusters[s], n_c[r], n_c[s]
        try:
            del e_in[r]
        except KeyError:
            pass
        try:
            del e_in[s]
        except KeyError:
            pass
        try:
            del e_out[r]
        except KeyError:
            pass
        try:
            del e_out[s]
        except KeyError:
            pass

        # Update sigmas
        nonlocal sigmas
        new_sigmas = {}
        for k in set(e_out.keys()).union(set(e_in.keys())):  # initialise sigmas
            new_sigmas[k] = 1
        update_sigmas_bt(new_sigmas, e_out, e_in)
        sigmas = new_sigmas

        # Update merges in ddl_dict
        if not exact:
            checked = []
            for u in all_rs_neigs:
                if u in pair:
                    continue
                for v in ddl_dict[u]:
                    if v in pair:
                        continue
                    if (u, v) in checked or (v, u) in checked:
                        pass
                    else:
                        relevant_terms_after_rs_merge = (
                            f((u, v), (r, s), sigma_dict[(u, v)], sigma_dict[(r, s)])
                            + f((r, s), (u, v), sigma_dict[(r, s)], sigma_dict[(u, v)])
                            - f(u, (r, s), sigma_dict[u], sigma_dict[(r, s)])
                            - f(v, (r, s), sigma_dict[v], sigma_dict[(r, s)])
                            - f((r, s), u, sigma_dict[(r, s)], sigma_dict[u])
                            - f((r, s), v, sigma_dict[(r, s)], sigma_dict[v])
                        )
                        relevant_terms_before_rs_merge = (
                            f((u, v), r, sigma_dict[(u, v)], sigma_dict[r])
                            + f(r, (u, v), sigma_dict[r], sigma_dict[(u, v)])
                            - f(u, r, sigma_dict[u], sigma_dict[r])
                            - f(v, r, sigma_dict[v], sigma_dict[r])
                            - f(r, u, sigma_dict[r], sigma_dict[u])
                            - f(r, v, sigma_dict[r], sigma_dict[v])
                            + f((u, v), s, sigma_dict[(u, v)], sigma_dict[s])
                            + f(s, (u, v), sigma_dict[s], sigma_dict[(u, v)])
                            - f(u, s, sigma_dict[u], sigma_dict[s])
                            - f(v, s, sigma_dict[v], sigma_dict[s])
                            - f(s, u, sigma_dict[s], sigma_dict[u])
                            - f(s, v, sigma_dict[s], sigma_dict[v])
                        )

                        ddl_dict[u][v] += (
                            relevant_terms_after_rs_merge - relevant_terms_before_rs_merge
                        )
                        ddl_dict[v][u] = ddl_dict[u][v]
                        checked.append((u, v))

    # Compute initial BT scores
    # print("Computing initial BT scores", file=stderr)
    update_sigmas_bt(sigmas, e_out, e_in)

    # Compute initial DL
    min_dl = dl = initial_dl = total_dl()
    bt_dl = initial_dl - loggamma(N + 1) - np.log(N)
    min_R = N
    min_sigmas = sigmas
    min_clusters = clusters
    print(f"Initial DL: {initial_dl}", file=stderr)
    print(f"Initial Ranks: {R}", file=stderr)
    # Print the number of workers
    if sync:
        cluster_resources = ray.cluster_resources()
        num_workers = int(cluster_resources.get("CPU", 0))
        print(f"Number of workers in the Pool: {num_workers}", file=stderr)
    print(f"Tolerance: {TARGET}", file=stderr)

    # Compute number of unique ranks inferred by BT
    BT_R = len(set(sigmas.values()))

    # If full trace, append resulst dictioanry to trace_list
    if full_trace:
        results_dict = {
            "N": N,  # Number of nodes
            "M": M,  # Number of edges
            "<k>": M / N,  # Average degree
            "R": R,  # Number of unique ranks
            "BT_R": BT_R,  # Number of unique ranks inferred by BT model
            "DL": dl,  # Description length
            "BT_DL": bt_dl,  # Description length of BT model
            "LPOR": bt_dl - dl,  # Log posterior odds ratio
            "CR": 1,  # Compression ratio
            "Strengths": sigmas,  # Strengths of each cluster
            "Clusters": deepcopy(clusters),  # Clusters
        }
        trace_list = [results_dict]

    iter_count = 0  # Initialise iteration counter

    # Main loop
    while True:
        start_time = time.time()
        iter_count += 1
        if verbose:
            print(f"Iteration {iter_count}", file=stderr)
        # Sort sigmas dictionary by value
        sorted_sigmas = dict(sorted(sigmas.items(), key=lambda item: item[1], reverse=False))
        # Define variables to store optimal values
        best_ddl = np.inf  # Best delta DL
        best_pair = None  # Best pair of clusters to merge
        ddl_dict = {}  # Dictionary to store delta DLs for all pairs
        sigma_dict = {}  # Dictionary to track sigmas for all pairs
        if sync:  # Use synchronous update
            # Create an array of adjacent pairs
            pairs = np.column_stack(
                (list(sorted_sigmas.keys())[:-1], list(sorted_sigmas.keys())[1:])
            )
            ddls = pool.map(worker, pairs)
            # Find the pair with the smallest ddl
            try:
                best_ddl = np.min([el[0] for el in ddls])
                best_pair = pairs[np.argmin([el[0] for el in ddls])]
                best_in_neigs = ddls[np.argmin([el[0] for el in ddls])][1]
                best_out_neigs = ddls[np.argmin([el[0] for el in ddls])][2]
            except ValueError:  # Avoid issues when all pairs have been merged
                best_ddl = np.inf
                best_pair = None
        else:
            # Iterate through adjacent pairs of keys
            for i in range(len(sorted_sigmas) - 1):
                # Select candidate pair of clusters to merge
                r, s = list(sorted_sigmas.keys())[i], list(sorted_sigmas.keys())[i + 1]
                # Compute delta DL, new sigmas, and new cluster label after merging r and s
                ddl, rs_in_neigs, rs_out_neigs = delta_dl(r, s)
                # Update best pair if delta DL is smaller than the current best
                if ddl < best_ddl:
                    best_ddl = ddl
                    best_pair = (r, s)
                    best_in_neigs = rs_in_neigs
                    best_out_neigs = rs_out_neigs

        # Add constant ddl term
        best_ddl += C(R - 1) - C(R)

        # Merge best pair
        try:
            if force_merge or best_ddl < 0:
                if verbose:
                    print(f"Merging: {best_pair}", file=stderr)
                # Merge ranks
                merge_ranks(best_pair, e_in, e_out, best_in_neigs, best_out_neigs)
                R -= 1
                if exact:
                    dl = total_dl()
                else:
                    dl += best_ddl
                # Update min_dl
                if dl < min_dl:
                    min_dl = dl
                    min_R = R
                    min_sigmas = sigmas
                    min_clusters = deepcopy(clusters)
                # If full trace, append results to trace_list
                if full_trace:
                    end_time = time.time()
                    results_dict = {
                        "N": N,
                        "M": M,
                        "<k>": M / N,
                        "R": R,
                        "BT_R": BT_R,
                        "DL": dl,
                        "BT_DL": bt_dl,
                        "LPOR": bt_dl - dl,
                        "CR": dl / initial_dl,
                        "Strengths": sigmas,
                        "Clusters": deepcopy(clusters),
                        "Time": end_time - start_time,
                    }
                    trace_list.append(results_dict)
                end_time = time.time()
                if verbose:
                    print(f"New DL: {dl}", file=stderr)
                    print(f"Time taken: {end_time - start_time}", file=stderr)
                # if iter_count == 10:
                #     break
            else:
                break
        except TypeError:  # If best_pair is None (happens when W is 1D)
            break

    # Print summary
    print(f"Converged in {iter_count} iterations", file=stderr)
    print(f"Partial Rankings: {min_R}", file=stderr)
    print(f"Initial DL: {initial_dl}", file=stderr)
    print(f"Min DL: {min_dl}", file=stderr)
    print(f"BT DL: {bt_dl}", file=stderr)
    print(f"LPOR: {bt_dl - min_dl}", file=stderr)
    print(f"CR: {min_dl / initial_dl}", file=stderr)

    if full_trace:
        return trace_list

    return {
        "N": N,
        "M": M,
        "<k>": M / N,
        "R": min_R,
        "BT_R": BT_R,
        "DL": min_dl,
        "BT_DL": bt_dl,
        "LPOR": bt_dl - min_dl,
        "CR": min_dl / initial_dl,
        "Strengths": min_sigmas,
        "Clusters": min_clusters,
    }

def get_N(match_list: np.ndarray) -> int:
    """
    Get the number of unique players in a match list

    Parameters
    ----------
    match_list : ndarray
        Array of matches of the form [[i, j],...] or [[i, j, w_ij],...] where w_ij is the number of times i beats j

    Returns
    -------
    N : int
        Number of unique players in the match list
    """

    # Get the number of unique players
    N = len(np.unique(match_list[:, :2]))
    return N


def get_M(match_list: np.ndarray, return_unique: bool = False) -> int:
    """
    Get the number of matches in a match list

    Parameters
    ----------
    match_list : ndarray
        Array of matches of the form [[i, j],...] or [[i, j, w_ij],...] where w_ij is the number of times i beats j
    return_unique : bool
        If True, return the number of unique matches


    Returns
    -------
    M : int
        Number of matches in the match list
    """

    # Get the number of columns in the match list
    num_cols = match_list.shape[1]

    # Get the number of matches
    M = np.sum([int(el[2]) for el in match_list]) if num_cols == 3 else len(match_list)

    if return_unique:
        # Get the number of unique matches
        if num_cols == 3:
            E = len(match_list)
        elif num_cols == 2:
            E = len(np.unique(match_list, axis=0))
        return M, E

    return M


def get_edges(match_list: np.ndarray) -> tuple:
    """
    Get the in and out edges from a match list

    Parameters
    ----------
    match_list : ndarray
        Array of matches of the form [[i, j],...] or [[i, j, w_ij],...] where w_ij is the number of times i beats j

    Returns
    -------
    e_out : dict
        Dictionary of dictionaries such that e_out[i][j] is the number of times i beats j
    e_in : dict
        Dictionary of dictionaries such that e_in[j][i] is the number of times j beats i
    """
    # Initialise dictionaries for in and out edges
    e_out = DefaultDict(dict)
    e_in = DefaultDict(dict)

    # Parse the match list
    for match in match_list:
        num_cols = len(match)  # Check for number of columns in data
        if num_cols == 2:
            i, j = match
            if i not in e_out:
                e_out[i] = DefaultDict(int)
            e_out[i][j] += 1
            if j not in e_in:
                e_in[j] = DefaultDict(int)
            e_in[j][i] += 1
        elif num_cols == 3:
            i, j, w = match
            if i not in e_out:
                e_out[i] = DefaultDict(int)
            e_out[i][j] += int(w)
            if j not in e_in:
                e_in[j] = DefaultDict(int)
            e_in[j][i] += int(w)

    return e_out, e_in
