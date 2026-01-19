import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

hydro_scale = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

in_file_template="""
read
    mol pqr {}
end

elec
    mg-auto
    dime 65 65 65
    cglen 80 80 80
    fglen 80 80 80
    cgcent mol 1
    fgcent mol 1
    mol 1
    lpbe
    bcfl mdh
    pdie 2.0
    sdie 78.54
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.3
    temp 298.15
    write pot dx tmp_pot.dx
end
quit
"""

def project_hbond_propensity(traj, vertices, mode, sigma=2.0):
    """Project H-bond donor or acceptor propensity as density to surface vertices using Gaussian kernel.
    :param traj: mdtraj.Trajectory object
    :param vertices: (M, 3) array of surface vertex coordinates
    :param mode: 'donor' or 'acceptor'
    :return: (M,) array of H-bond propensity at each vertex
    """
    topology = traj.top
    atom_pos = traj.xyz[0]

    # Identify donor or acceptor atoms (rough heuristic based on atom types)
    hb_atoms = []
    for atom in topology.atoms:
        elem = atom.element.symbol
        if mode == 'donor':
            # Atoms that can donate: N or O with attached H (check for H in residue)
            if elem in ['N', 'O'] and any(a.element.symbol == 'H' for a in atom.residue.atoms if a != atom):
                hb_atoms.append(atom.index)
        elif mode == 'acceptor':
            # Atoms that can accept: O, N, S
            if elem in ['O', 'N', 'S']:
                hb_atoms.append(atom.index)

    if not hb_atoms:
        return np.zeros(len(vertices))

    hb_pos = atom_pos[hb_atoms]

    # Compute distances, this is euclidian but that's fine because we are dealing with close vertices in the mesh/graph
    dists = cdist(vertices, hb_pos)
    weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

    # Propensity as summed density
    propensity = np.sum(weights, axis=1)

    # Normalize by max for relative scale (optional)
    if np.max(propensity) > 0:
        propensity /= np.max(propensity)

    return propensity


def project_hydrophobicity(traj, vertices):
    """Project hydrophobicity to surface vertices using inverse distance weighting from residue scales.
    :param traj: mdtraj.Trajectory object
    :param vertices: (M, 3) array of surface vertex coordinates
    :return: (M,) array of hydrophobicity at each vertex
    """
    topology = traj.top
    atom_pos = traj.xyz[0]

    # Assign hydrophobicity to each atom based on its residue
    atom_hydro = np.array([hydro_scale.get(atom.residue.name, 0.0) for atom in topology.atoms])

    # Compute distances from vertices to all atoms
    dists = cdist(vertices, atom_pos)

    # Inverse distance weighting (power=2)
    epsilon = 1e-3
    weights = 1.0 / (dists + epsilon) ** 2

    # Weighted average
    hydro_vert = np.sum(weights * atom_hydro[None, :], axis=1) / np.sum(weights, axis=1)

    return hydro_vert

def compute_geodesic_distances(vertices, faces):
    """
    Compute geodesic distance matrix between all vertices on the mesh using Dijkstra's algorithm.
    :param vertices: (N, 3) array of vertex positions
    :param faces: (M, 3) array of triangle vertex indices
    :return: (N, N) array of geodesic distances
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    V = vertices.shape[0]
    F = faces.shape[0]

    edges = np.vstack([
        faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]
    ])
    weights = np.linalg.norm(vertices[edges[:,0]] - vertices[edges[:,1]], axis=1)

    # Build CSR matrix
    src = np.concatenate([edges[:,0], edges[:,1]])
    dst = np.concatenate([edges[:,1], edges[:,0]])
    w   = np.concatenate([weights, weights])

    # Sort by src
    order = np.argsort(src)
    src, dst, w = src[order], dst[order], w[order]

    # indptr
    indptr = np.zeros(V + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=V))

    # CSR
    csr = csr_matrix((w, dst, indptr), shape=(V, V))

    # Dijkstra
    dist = dijkstra(csr, directed=False, return_predecessors=False)
    np.fill_diagonal(dist, 0.0)
    return dist

def get_patch_residues(traj, atoms):
    """
    Because we are also collencting atom ids in each patch we can go back and get the residuies, this is mostly for
    testing and visualizaion which will become importatnt later
    :param atoms: the atoms of the patch
    :return: a dict that looks like
    {'chain':chain_id, 'residue': [list of residue numbers]}
    """

    unique_atom_indices = np.unique(atoms)
    unique_atom_indices = unique_atom_indices[unique_atom_indices >= 0]  # Filter invalid

    residues = set()
    for ai in unique_atom_indices:
        atom = traj.top.atom(ai)
        residues.add(atom.residue)

    # Sort by residue index
    res_list = sorted(residues, key=lambda r: r.index)

    # Format as "Chain C Res 123ALA"
    residues = [{"chain":r.chain.index,
                 "residue": r.resSeq} for r in res_list]

    return residues

#TODO need to reconstruct patches pcd and fpfh from data
def load_patches(file):
    pass