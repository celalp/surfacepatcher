import numpy as np

import subprocess
import mdtraj as md
import open3d as o3d

from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

hydro_scale = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

def compute_msms_surface(coords, traj):
    """
    Compute molecular surface using MSMS and return vertices, faces, normals, and atom IDs.
    :param coords: (N, 3) array of atomic coordinates
    :param traj: mdtraj.Trajectory object with topology information
    :return: vertices (M, 3), faces (K, 3), normals (M, 3), atom_ids (M,)
    """
    # Write temporary PDB for MSMS
    tmp_pdb = "tmp.pdb"
    tmp_traj = md.Trajectory(coords, traj.top)
    tmp_traj.save_pdb(tmp_pdb)

    # Run MSMS (requires MSMS binary)
    with open("tmp.xyzr", "w") as out:
        results=subprocess.run(
            ["pdb_to_xyzr", tmp_pdb],
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    subprocess.run([
        "msms", "-if", "tmp.xyzr", "-probe_radius", "1.4",
        "-density", "2.0", "-of", "tmp"  # Produces tmp.vert and tmp.face
    ])

    # Load MSMS output (.vert has 3 header lines)
    vert_data = np.loadtxt("tmp.vert", skiprows=3)
    vertices = vert_data[:, 0:3]  # x y z
    normals = vert_data[:, 3:6]  # nx ny nz
    atom_ids = vert_data[:, 7].astype(int) - 1  # closest atom index (1-based to 0-based)

    # .face has 3 headers
    face_data = np.loadtxt("tmp.face", skiprows=3)
    faces = face_data[:, 0:3].astype(int) - 1  # 1-based to 0-based

    return vertices, faces, normals, atom_ids


def project_electrostatics(traj, vertices):
    """Project approximate electrostatic potential to surface vertices using simple
    Coulomb summation from charged atoms. The inputs are from compute_msms_surface method
    :param traj: mdtraj.Trajectory object
    :param vertices: (M, 3) array of surface vertex coordinates
    :return: (M,) array of electrostatic potential at each vertex
    """
    topology = traj.top
    atom_pos = traj.xyz[0]  # Assuming single frame

    # Identify charged atoms and their charges (simplified: distribute residue charge over sidechain atoms)
    charged_atoms = []
    charges = []
    for residue in topology.residues:
        res_name = residue.name
        if res_name in ['ARG', 'LYS']:
            total_charge = 1.0
            sidechain_atoms = []
            if res_name == 'ARG':
                sidechain_atoms = ['NH1', 'NH2', 'CZ']  # Distribute over guanidino group
            elif res_name == 'LYS':
                sidechain_atoms = ['NZ']
            for atom in residue.atoms:
                if atom.name in sidechain_atoms:
                    charged_atoms.append(atom.index)
                    charges.append(total_charge / len(sidechain_atoms))
        elif res_name in ['GLU', 'ASP']:
            total_charge = -1.0
            sidechain_atoms = []
            if res_name == 'GLU':
                sidechain_atoms = ['OE1', 'OE2']
            elif res_name == 'ASP':
                sidechain_atoms = ['OD1', 'OD2']
            for atom in residue.atoms:
                if atom.name in sidechain_atoms:
                    charged_atoms.append(atom.index)
                    charges.append(total_charge / len(sidechain_atoms))
        # Note: Ignoring HIS, other titratable groups for simplicity

    if not charged_atoms:
        return np.zeros(len(vertices))  # No charges, zero potential

    charged_pos = atom_pos[charged_atoms]
    charges = np.array(charges)

    # Compute distances from vertices to charged atoms
    dists = cdist(vertices, charged_pos)

    # Coulomb potential: sum (q / r), with epsilon to avoid division by zero
    epsilon = 1e-3  # Increased epsilon for more stability
    potentials = np.sum(charges / (dists + epsilon), axis=1)
    # Clamp extreme electrostatic values
    potentials = np.clip(potentials, -1e3, 1e3)

    return potentials

def project_hbond_propensity(traj, vertices, mode):
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

    # Compute distances
    dists = cdist(vertices, hb_pos)

    # Gaussian kernel for density (sigma=2.0 Å, adjustable)
    sigma = 2.0
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


# Helper functions for curvature and geodesic
def triangle_area(p0, p1, p2):
    """Return area of a triangle given three points."""
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))


def cotangent_weight(p0, p1, p2):
    """
    Compute cotangent of the angle at p0 in triangle (p0, p1, p2).
    Returns cot(alpha) where alpha is the angle at p0.
    """
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p0 - p2)
    c = np.linalg.norm(p0 - p1)
    # cot(alpha) = (b^2 + c^2 - a^2) / (4 * area)
    area = triangle_area(p0, p1, p2)
    # Use robust epsilon to avoid division by zero
    epsilon = 1e-8
    if area < epsilon:
        return 0.0
    cot = (b ** 2 + c ** 2 - a ** 2) / (4 * area)
    # Clamp extreme values to prevent inf propagation
    return np.clip(cot, -1e6, 1e6)

def compute_curvature(vertices, faces):
    """
    Compute mean curvature and shape index for each vertex on the mesh.
    Returns dict with 'mean_curvature': (N,) signed mean H, 'shape_index': (N,) in [-1,1]
    :param vertices: (N, 3) array of vertex positions
    :param faces: (M, 3) array of triangle vertex indices
    :return: mean_curvature (N,), shape_index (N,)
    """
    V = vertices
    F = faces
    N = V.shape[0]

    # Compute vertex normals using Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # Precompute adjacency and cotangents
    adj = [[] for _ in range(N)]  # list of (j, cot_alpha + cot_beta for edge ij)
    incident_tri = [[] for _ in range(N)]  # list of triangle indices incident to i

    tri_areas = np.zeros(len(F))
    for t, (a, b, c) in enumerate(F):
        tri_areas[t] = triangle_area(V[a], V[b], V[c])
        incident_tri[a].append(t)
        incident_tri[b].append(t)
        incident_tri[c].append(t)

    # Barycentric areas A
    A = np.zeros(N)
    for i in range(N):
        A[i] = (1.0 / 3.0) * np.sum(tri_areas[incident_tri[i]])

    # Compute cotangents for Laplace
    for t, (a, b, c) in enumerate(F):
        cot_a = cotangent_weight(V[a], V[b], V[c])
        cot_b = cotangent_weight(V[b], V[c], V[a])
        cot_c = cotangent_weight(V[c], V[a], V[b])

        # For edge ab: cot_c (opposite)
        adj[a].append((b, cot_c))
        adj[b].append((a, cot_c))
        # For edge bc: cot_a
        adj[b].append((c, cot_a))
        adj[c].append((b, cot_a))
        # For edge ca: cot_b
        adj[c].append((a, cot_b))
        adj[a].append((c, cot_b))

    # Build cotangent Laplacian L (symmetric, off-diag negative)
    L = lil_matrix((N, N))
    for i in range(N):
        sum_w = 0.0
        for j, w in adj[i]:
            L[i, j] -= w  # off-diagonal -w
            sum_w += w
        L[i, i] = sum_w  # diagonal sum w

    L = L.tocsr()

    # Mass matrix inverse with robust epsilon
    epsilon_area = 1e-6
    safe_areas = np.maximum(2 * A, epsilon_area)
    Minv = diags(1.0 / safe_areas, 0)

    # Mean curvature normal operator
    Delta_V = Minv @ (-L @ V)  # shape (N,3)
    HN = - Delta_V
    H_abs = np.linalg.norm(HN, axis=1) / 2.0
    # Clamp extreme curvature values
    H_abs = np.clip(H_abs, 0, 1e3)
    sign = np.sign(np.sum(normals * HN, axis=1))
    mean_curvature = sign * H_abs

    # For Gaussian curvature
    kg = np.zeros(N)
    epsilon_area = 1e-6
    for i in range(N):
        sum_theta = 0.0
        for t in incident_tri[i]:
            a, b, c = F[t]
            # Find the angle at i
            if a == i:
                p0, p1, p2 = V[a], V[b], V[c]
            elif b == i:
                p0, p1, p2 = V[b], V[c], V[a]
            else:
                p0, p1, p2 = V[c], V[a], V[b]
            cot = cotangent_weight(p0, p1, p2)
            # Robust angle calculation
            if abs(cot) < 1e-8:
                theta = np.pi / 2
            else:
                theta = np.arctan(1 / cot)
            sum_theta += theta
        # Use robust epsilon and clamp result
        safe_area = max(A[i], epsilon_area)
        kg[i] = np.clip((2 * np.pi - sum_theta) / safe_area, -1e3, 1e3)

    # Principal curvatures
    disc = mean_curvature ** 2 - kg
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    k1 = mean_curvature + sqrt_disc
    k2 = mean_curvature - sqrt_disc

    # Shape index with robust computation
    k_max = np.maximum(k1, k2)
    k_min = np.minimum(k1, k2)
    epsilon_shape = 1e-6
    denom = np.maximum(k_max - k_min, epsilon_shape)
    ratio = (k_max + k_min) / denom
    # Clamp ratio to prevent extreme arctan values
    ratio = np.clip(ratio, -1e3, 1e3)
    shape_index = (2 / np.pi) * np.arctan(ratio)
    # Final safety: ensure shape_index is in valid range
    shape_index = np.clip(shape_index, -1.0, 1.0)

    return mean_curvature, shape_index

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

