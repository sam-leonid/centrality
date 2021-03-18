#    Copyright (C) 2004-2018 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    Richard Penney <rwpenney@users.sourceforge.net>
#    All rights reserved.
#    BSD license.
#
# Authors: Aric Hagberg <aric.hagberg@gmail.com>,
#          Dan Schult <dschult@colgate.edu>
"""
******
Layout
******

Node positioning algorithms for graph drawing.

For `random_layout()` the possible resulting shape
is a square of side [0, scale] (default: [0, 1])
Changing `center` shifts the layout by that amount.

For the other layout routines, the extent is
[center - scale, center + scale] (default: [-1, 1]).

Warning: Most layout routines have only been tested in 2-dimensions.

"""
from __future__ import division
import networkx as nx

__all__ = ['bipartite_layout',
           'circular_layout',
           'kamada_kawai_layout',
           'random_layout',
           'rescale_layout',
           'shell_layout',
           'spring_layout',
           'spectral_layout',
           'fruchterman_reingold_layout']


def _process_params(G, center, dim):
    # Some boilerplate code.
    import numpy as np

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center



def circular_layout(G, scale=1, center=None, dim=2):
    # dim=2 only
    """Position nodes on a circle.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.
        If dim>2, the remaining dimensions are set to zero
        in the returned positions.
        If dim<2, a ValueError is raised.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    -------
    ValueError
        If dim < 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.circular_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    if dim < 2:
        raise ValueError('cannot handle dimensions < 2')

    G, center = _process_params(G, center, dim)

    paddims = max(0, (dim - 2))

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {nx.utils.arbitrary_element(G): center}
    else:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(G) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta), np.sin(theta),
                               np.zeros((len(G), paddims))])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos



def shell_layout(G, nlist=None, scale=1, center=None, dim=2):
    """Position nodes in concentric circles.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nlist : list of lists
       List of node lists for each shell.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout, currently only dim=2 is supported.
        Other dimension values result in a ValueError.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Raises
    -------
    ValueError
        If dim != 2

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> shells = [[0], [1, 2, 3]]
    >>> pos = nx.shell_layout(G, shells)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    if dim != 2:
        raise ValueError('can only handle 2 dimensions')

    G, center = _process_params(G, center, dim)

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {nx.utils.arbitrary_element(G): center}

    if nlist is None:
        # draw the whole graph in one shell
        nlist = [list(G)]

    if len(nlist[0]) == 1:
        # single node at center
        radius = 0.0
    else:
        # else start at r=1
        radius = 1.0

    npos = {}
    for nodes in nlist:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(nodes) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta), np.sin(theta)])
        pos = rescale_layout(pos, scale=scale * radius / len(nlist)) + center
        npos.update(zip(nodes, pos))
        radius += 1.0

    return npos



def bipartite_layout(G, nodes, align='vertical',
                     scale=1, center=None, aspect_ratio=4/3):
    """Position nodes in two straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    nodes : list or container
        Nodes in one node set of the bipartite graph.
        This set will be placed on left or top.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    aspect_ratio : number (default=4/3):
        The ratio of the width to the height of the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.bipartite.gnmk_random_graph(3, 5, 10, seed=123)
    >>> top = nx.bipartite.sets(G)[0]
    >>> pos = nx.bipartite_layout(G, top)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """

    import numpy as np

    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    height = 1
    width = aspect_ratio * height
    offset = (width/2, height/2)

    top = set(nodes)
    bottom = set(G) - top
    nodes = list(top) + list(bottom)

    if align == 'vertical':
        left_xs = np.repeat(0, len(top))
        right_xs = np.repeat(width, len(bottom))
        left_ys = np.linspace(0, height, len(top))
        right_ys = np.linspace(0, height, len(bottom))

        top_pos = np.column_stack([left_xs, left_ys]) - offset
        bottom_pos = np.column_stack([right_xs, right_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    if align == 'horizontal':
        top_ys = np.repeat(height, len(top))
        bottom_ys = np.repeat(0, len(bottom))
        top_xs = np.linspace(0, width, len(top))
        bottom_xs = np.linspace(0, width, len(bottom))

        top_pos = np.column_stack([top_xs, top_ys]) - offset
        bottom_pos = np.column_stack([bottom_xs, bottom_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    msg = 'align must be either vertical or horizontal.'
    raise ValueError(msg)

def kamada_kawai_layout(G, dist=None,
                        pos=None,
                        weight='weight',
                        scale=1,
                        center=None,
                        dim=2):
    """Position nodes using Kamada-Kawai path-length cost-function.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    dist : float (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.kamada_kawai_layout(G)
    """
    import numpy as np

    G, center = _process_params(G, center, dim)
    nNodes = len(G)

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = {n: pt for n, pt in zip(G, np.linspace(0, 1, len(G)))}
    pos_arr = np.array([pos[n] for n in G])

    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim)

    pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(G, pos))



def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np
    from scipy.optimize import minimize

    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3),
                meanwt, dim)

    optresult = minimize(_kamada_kawai_costfn, pos_arr.ravel(),
                         method='L-BFGS-B', args=costargs, jac=True)

    return optresult.x.reshape((-1, dim))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum('ijk,ij->ijk',
                          delta,
                          1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset ** 2)
    grad = (np.einsum('ij,ij,ijk->ik', invdist, offset, direction) -
            np.einsum('ij,ij,ijk->jk', invdist, offset, direction))

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos ** 2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())

def spectral_layout(G, weight='weight', scale=1, center=None, dim=2):
    """Position nodes using the eigenvectors of the graph Laplacian.

    Using the unnormalized Laplacion, the layout shows possible clusters of
    nodes which are an approximation of the ratio cut. If dim is the number of
    dimensions then the positions are the entries of the dim eigenvectors
    corresponding to the ascending eigenvalues starting from the second one.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> pos = nx.spectral_layout(G)

    Notes
    -----
    Directed graphs will be considered as undirected graphs when
    positioning the nodes.

    For larger graphs (>500 nodes) this will use the SciPy sparse
    eigenvalue solver (ARPACK).
    """
    # handle some special cases that break the eigensolvers
    import numpy as np

    G, center = _process_params(G, center, dim)

    if len(G) <= 2:
        if len(G) == 0:
            pos = np.array([])
        elif len(G) == 1:
            pos = np.array([center])
        else:
            pos = np.array([np.zeros(dim), np.array(center) * 2.0])
        return dict(zip(G, pos))
    try:
        # Sparse matrix
        if len(G) < 500:  # dense solver is faster for small graphs
            raise ValueError
        A = nx.to_scipy_sparse_matrix(G, weight=weight, dtype='d')
        # Symmetrize directed graphs
        if G.is_directed():
            A = A + np.transpose(A)
        pos = _sparse_spectral(A, dim)
    except (ImportError, ValueError):
        # Dense matrix
        A = nx.to_numpy_array(G, weight=weight)
        # Symmetrize directed graphs
        if G.is_directed():
            A += A.T
        pos = _spectral(A, dim)

    pos = rescale_layout(pos, scale) + center
    pos = dict(zip(G, pos))
    return pos



def _spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses dense eigenvalue solver from numpy
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError:
        msg = "spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg)

    # form Laplacian matrix
    I = np.identity(nnodes, dtype=A.dtype)
    D = I * np.sum(A, axis=1)  # diagonal of degrees
    L = D - A

    eigenvalues, eigenvectors = np.linalg.eig(L)
    # sort and keep smallest nonzero
    index = np.argsort(eigenvalues)[1:dim + 1]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])


def _sparse_spectral(A, dim=2):
    # Input adjacency matrix A
    # Uses sparse eigenvalue solver from scipy
    # Could use multilevel methods here, see Koren "On spectral graph drawing"
    import numpy as np
    from scipy.sparse import spdiags
    from scipy.sparse.linalg.eigen import eigsh

    try:
        nnodes, _ = A.shape
    except AttributeError:
        msg = "sparse_spectral() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg)

    # form Laplacian matrix
    data = np.asarray(A.sum(axis=1).T)
    D = spdiags(data, 0, nnodes, nnodes)
    L = D - A

    k = dim + 1
    # number of Lanczos vectors for ARPACK solver.What is the right scaling?
    ncv = max(2 * k + 1, int(np.sqrt(nnodes)))
    # return smallest k eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k, which='SM', ncv=ncv)
    index = np.argsort(eigenvalues)[1:k]  # 0 index is zero eigenvalue
    return np.real(eigenvectors[:, index])


def rescale_layout(pos, scale=1):
    """Returns scaled position array to (-scale, scale) in all axes.

    The function acts on NumPy arrays which hold position information.
    Each position is one row of the array. The dimension of the space
    equals the number of columns. Each coordinate in one column.

    To rescale, the mean (center) is subtracted from each axis separately.
    Then all values are scaled so that the largest magnitude value
    from all axes equals `scale` (thus, the aspect ratio is preserved).
    The resulting NumPy Array is returned (order of rows unchanged).

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number (default: 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.

    """
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos



# fixture for nose tests
def setup_module(module):
    from nose import SkipTest
    try:
        import numpy
    except:
        raise SkipTest("NumPy not available")
    try:
        import scipy
    except:
        raise SkipTest("SciPy not available")