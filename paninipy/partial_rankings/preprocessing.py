"""
Module containing functions for preprocessing match lists.
"""

import numpy as np


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
