"""
Module containing the code for the greedy regionalization algorithm
"""

import numpy as np
from scipy.special import loggamma


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


def logchoose(N, K):
    """
    computes log of binomial coefficient

    Parameters
    ----------
    N : int
        total number of elements
    K : int
        number of elements to choose

    Returns
    -------
    float
        log of binomial coefficient
    """
    if (K <= 0) or (N <= 0):
        return 0
    return loggamma(N + 1) - loggamma(K + 1) - loggamma(N - K + 1)


def logmultiset(N, K):
    """
    computes log of multiset coefficient

    Parameters
    ----------
    N : int
        total number of elements
    K : int
        number of elements to choose

    Returns
    -------
    float
        log of multiset coefficient
    """
    return logchoose(N + K - 1, K)


def greedy_opt(N, spatial_elist, flow_elist):
    """
    fast greedy regionalization for objective functions of the form:

                C(B) + sum_r g(r) + sum_rs f(r,s),

    where r,s index clusters and B is the number of clusters.

    Parameters
    ----------
    N : int
        number of nodes
    spatial_elist : list of tuples
        list of edges (i,j) defined by the spatial adjacency between i and j (no repeats)
    flow_elist : list of tuples
        list of weighted edges defined by flows (i,j,w), where flow is from i --> j and has weight w (no repeats)

    Returns
    -------
    DLs : list of floats
        list of description length values at each iteration
    partitions : list of lists
        list of partitions at each iteration

    Notes
    -----
    Make sure nodes are indexed as 0,....,N-1 so as to handle nodes with no flows.
    """

    E = len(flow_elist)  # number of edges
    W = sum([e[-1] for e in flow_elist])  # total flow
    B = N  # initial number of clusters

    clusters, n_c = {}, {}  # dictionaries to track clusters and their sizes
    for i in range(N):
        clusters[i] = set([i])
        n_c[i] = 1

    # directed dictionaries for the number of edges between clusters and total flow between clusters
    # For example:
    #           e_out[i][j] is the number of edges going out from cluster i to cluster j
    #           e_in[i][j] is the number  edges coming into cluster i from cluster j
    # Both dictionaries are needed since we sometimes iterate over in-neighbors, and sometimes over out-neighbors
    ein_c, win_c, eout_c, wout_c = (
        DefaultDict(dict),
        DefaultDict(dict),
        DefaultDict(dict),
        DefaultDict(dict),
    )
    for i in range(N):
        ein_c[i], win_c[i], eout_c[i], wout_c[i] = (
            DefaultDict(int),
            DefaultDict(int),
            DefaultDict(int),
            DefaultDict(int),
        )

    for e in flow_elist:

        i, j, w = e

        eout_c[i][j] = 1
        wout_c[i][j] = w
        ein_c[j][i] = 1
        win_c[j][i] = w

    # function definitions for C, g, and f. Can be changed depending on the clustering objective of interest
    def C(B):
        """
        Global contribution to the description length

        Parameters
        ----------
        B : int
            number of clusters

        Returns
        -------
        float
            global contribution to the description length
        """
        return (
            logmultiset(B**2, E)
            + logchoose(N - 1, B - 1)
            + logmultiset(B**2, W)
            + np.log(N)
            + loggamma(N + 1)
        )

    def g(r):
        """
        Computes the cluster-level contribution to the description length for
        cluster r. If a tuple is entered as the cluster index, it adds the
        corresponding terms for those indices for the merge.

        Parameters
        ----------
        r : int or tuple
            cluster index or tuple of cluster indices

        Returns
        -------
        float
            cluster-level contribution to the description length
        """

        if isinstance(r, tuple):
            n_r = n_c[r[0]] + n_c[r[1]]
        else:
            n_r = n_c[r]

        # return -N*np.log(n_r/N) #use stirling approximation of log binomial
        return -loggamma(n_r + 1)

    def f(r, s):
        """
        Cluster-to-cluster contribution to the description length.
        Computes the term for r --> s.
        If a tuple is entered as the cluster index, it adds the corresponding
        terms for those indices for the merge.

        Parameters
        ----------
        r : int or tuple
            cluster index or tuple of cluster indices
        s : int or tuple
            cluster index or tuple of cluster indices

        Returns
        -------
        float
            cluster-to-cluster contribution to the description length
        """
        if isinstance(r, tuple) and isinstance(s, tuple):
            n_r = n_c[r[0]] + n_c[r[1]]
            n_s = n_c[s[0]] + n_c[s[1]]
            e_rs = (
                eout_c[r[0]][s[0]]
                + eout_c[r[0]][s[1]]
                + eout_c[r[1]][s[0]]
                + eout_c[r[1]][s[1]]
            )
            w_rs = (
                wout_c[r[0]][s[0]]
                + wout_c[r[0]][s[1]]
                + wout_c[r[1]][s[0]]
                + wout_c[r[1]][s[1]]
            )

        elif isinstance(r, tuple):
            n_r = n_c[r[0]] + n_c[r[1]]
            n_s = n_c[s]
            e_rs = eout_c[r[0]][s] + eout_c[r[1]][s]
            w_rs = wout_c[r[0]][s] + wout_c[r[1]][s]

        elif isinstance(s, tuple):
            n_r = n_c[r]
            n_s = n_c[s[0]] + n_c[s[1]]
            e_rs = eout_c[r][s[0]] + eout_c[r][s[1]]
            w_rs = wout_c[r][s[0]] + wout_c[r][s[1]]

        else:
            n_r = n_c[r]
            n_s = n_c[s]
            e_rs = eout_c[r][s]
            w_rs = wout_c[r][s]

        return logchoose(n_r * n_s, e_rs) + logchoose(w_rs - 1, e_rs - 1)

    def total_dl():
        """
        Computes the total description length or objective value of interest

        Returns
        -------
        float
            total description length
        """
        dl = C(B) + sum([g(r) for r in n_c])
        for r in eout_c:
            for s in eout_c[r]:
                dl += f(r, s)

        return dl

    def delta_dl(r, s):
        """
        Computes the change in description length after merging clusters r and s.
        Only needs to be computed entirely when the merge (r,s) is not stored
        in the ddl_c dictionary used to track merges.

        Parameters
        ----------
        r : int
            cluster index
        s : int
            cluster index

        Returns
        -------
        total_change : float
            total change in description length after merging clusters r and s
        """

        if r in ddl_c:
            if s in ddl_c[r]:
                return ddl_c[r][s]

        # cluster-level change in the description length
        dg = g((r, s)) - g(r) - g(s)

        # compute in and out neighbors of the merged cluster
        r_in_neigs, r_out_neigs, s_in_neigs, s_out_neigs = (
            set(ein_c[r].keys()),
            set(eout_c[r].keys()),
            set(ein_c[s].keys()),
            set(eout_c[s].keys()),
        )
        rs_in_neigs = r_in_neigs.union(s_in_neigs) - set([r, s])
        rs_out_neigs = r_out_neigs.union(s_out_neigs) - set([r, s])
        rs_all_neigs = rs_in_neigs.union(rs_out_neigs)

        # changes from neighboring nodes
        df_external = 0
        for u in rs_all_neigs:
            df_external += (
                f((r, s), u) + f(u, (r, s)) - f(r, u) - f(s, u) - f(u, r) - f(u, s)
            )

        # change from flows from r to s
        df_internal = f((r, s), (r, s)) - f(r, s) - f(s, r) - f(r, r) - f(s, s)

        # compute the total change in the description length
        total_change = dg + df_external + df_internal

        # store the change in description length in the ddl_c dictionary
        if not (r in ddl_c):
            ddl_c[r] = {}
        if not (s in ddl_c):
            ddl_c[s] = {}
        ddl_c[r][s] = total_change
        ddl_c[s][r] = total_change

        return total_change

    def merge_updates(r, s, DL):
        """
        Merges clusters r and s and updates the ddl_c dictionary of changes in
        description length for all nodes with flows into or out of r or s.
        r and s are the two clusters with best description length change after
        checking all possible merges. Deletes all obsolete information to save memory.

        Parameters
        ----------
        r : int
            cluster index
        s : int
            cluster index
        DL : float
            current description length

        Returns
        -------
        DL : float
            updated description length
        """

        # initialize a new key for the merged cluster key
        rs = np.random.randint(100000000)

        # compute the in and out neighbors of the merged cluster
        r_in_neigs, r_out_neigs, s_in_neigs, s_out_neigs = (
            set(ein_c[r].keys()),
            set(eout_c[r].keys()),
            set(ein_c[s].keys()),
            set(eout_c[s].keys()),
        )
        rs_in_neigs = r_in_neigs.union(s_in_neigs) - set([r, s])
        rs_out_neigs = r_out_neigs.union(s_out_neigs) - set([r, s])
        all_rs_neigs = rs_in_neigs.union(rs_out_neigs)

        # update the dictionaries for the clusters and their sizes
        clusters[rs] = clusters[r].union(clusters[s])
        n_c[rs] = n_c[r] + n_c[s]

        # create a new entry for the merged cluster in the dictionaries for the number of edges and total flow
        ein_c[rs], eout_c[rs], win_c[rs], wout_c[rs] = (
            DefaultDict(int),
            DefaultDict(int),
            DefaultDict(int),
            DefaultDict(int),
        )
        # contributions from neighbors of merged cluster
        for u in all_rs_neigs:

            ein_c[rs][u] = ein_c[r][u] + ein_c[s][u]
            win_c[rs][u] = win_c[r][u] + win_c[s][u]
            ein_c[u][rs] = ein_c[u][r] + ein_c[u][s]
            win_c[u][rs] = win_c[u][r] + win_c[u][s]

            eout_c[rs][u] = eout_c[r][u] + eout_c[s][u]
            wout_c[rs][u] = wout_c[r][u] + wout_c[s][u]
            eout_c[u][rs] = eout_c[u][r] + eout_c[u][s]
            wout_c[u][rs] = wout_c[u][r] + wout_c[u][s]

        # contribution from the clusters being merged
        ein_c[rs][rs] = ein_c[r][r] + ein_c[r][s] + ein_c[s][r] + ein_c[s][s]
        win_c[rs][rs] = win_c[r][r] + win_c[r][s] + win_c[s][r] + win_c[s][s]
        eout_c[rs][rs] = eout_c[r][r] + eout_c[r][s] + eout_c[s][r] + eout_c[s][s]
        wout_c[rs][rs] = wout_c[r][r] + wout_c[r][s] + wout_c[s][r] + wout_c[s][s]

        # remove references to old clusters from the ddl_c dictionary
        rs_ddls = set(ddl_c[r].keys()).union(set(ddl_c[s].keys())) - set([r, s])
        for u in rs_ddls:
            ddl_c[u].pop(r, None)
            ddl_c[r].pop(u, None)
            ddl_c[u].pop(s, None)
            ddl_c[s].pop(u, None)

        # update past merges that involve the (flow) neighbors of the merged cluster
        checked = []
        for u in all_rs_neigs:
            try:
                for v in ddl_c[u]:
                    if (u, v) in checked or (v, u) in checked:
                        pass
                    else:
                        relevant_terms_after_rs_merge = (
                            f((u, v), (r, s))
                            + f((r, s), (u, v))
                            - f(u, (r, s))
                            - f(v, (r, s))
                            - f((r, s), u)
                            - f((r, s), v)
                        )
                        relevant_terms_before_rs_merge = (
                            f((u, v), r)
                            + f(r, (u, v))
                            - f(u, r)
                            - f(v, r)
                            - f(r, u)
                            - f(r, v)
                            + f((u, v), s)
                            + f(s, (u, v))
                            - f(u, s)
                            - f(v, s)
                            - f(s, u)
                            - f(s, v)
                        )

                        ddl_c[u][v] += (
                            relevant_terms_after_rs_merge
                            - relevant_terms_before_rs_merge
                        )
                        ddl_c[v][u] = ddl_c[u][v]
                        checked.append((u, v))
            except KeyError:
                pass

        # include the global-level change and update the description length
        DL += ddl_c[r][s] + C(B - 1) - C(B)

        # remove references to old clusters from the neighbors' dictionaries
        for u in all_rs_neigs:
            ein_c[u].pop(r, None)
            ein_c[u].pop(s, None)
            win_c[u].pop(r, None)
            win_c[u].pop(s, None)
            eout_c[u].pop(r, None)
            eout_c[u].pop(s, None)
            wout_c[u].pop(r, None)
            wout_c[u].pop(s, None)

        for u in rs_ddls:
            ddl_c[rs][u] = delta_dl(rs, u)
            ddl_c[u][rs] = ddl_c[rs][u]

        # clean up: remove obsolete entries in neighbors' dictionaries and remove all entries correponding to r and s
        del (
            clusters[r],
            clusters[s],
            n_c[r],
            n_c[s],
            ein_c[r],
            ein_c[s],
            eout_c[r],
            eout_c[s],
            win_c[r],
            win_c[s],
            wout_c[r],
            wout_c[s],
            ddl_c[r],
            ddl_c[s],
        )

        return DL

    ddl_c = (
        {}
    )  # dictionary for past merges, indexed by only spatially adjacent neighbors
    for e in spatial_elist:

        i, j = e
        ddl_c[i][j] = delta_dl(i, j)
        ddl_c[j][i] = ddl_c[i][j]

    # Initialize the description length and the list of partitions
    DL = total_dl()
    DLs, partitions = [DL], [clusters.copy()]

    # Iterate over the number of clusters and find the best pair to merge
    for B in range(N, 1, -1):
        best_pair = (np.inf, np.inf)
        best_ddl = np.inf
        for i in ddl_c:
            for j in ddl_c[i]:
                if i < j:
                    ddl = delta_dl(i, j)
                    if ddl < best_ddl:
                        best_ddl = ddl
                        best_pair = (i, j)

        r, s = best_pair
        if r == s == np.inf:  # If no more merges are possible return the partitions
            return DLs, partitions

        # Merge the best pair and update the description length
        DL = merge_updates(r, s, DL)

        # Uncomment to stop algorithm when the description length increases
        if DL > DLs[-1]:
            return DLs, partitions

        # Store the description length and the partition
        DLs.append(DL)
        partitions.append(list(clusters.copy().values()))

    return DLs, partitions
