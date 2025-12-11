import os
import subprocess
import numpy as np
import mdtraj as md

from surfacepatcher.utils import in_file_template

class PDBProcessor:
    def __init__(self, pdb_file, abps_bin="apbs", multivalue_bin="multivalue"):
        self.pdb_file = pdb_file
        self.abps_bin = abps_bin
        self.multivalue_bin = multivalue_bin


    def _run_reduce(self, reduced_pdb_file="tmp_reduced.pdb"):
        """
        run reduce to add hydrogens to a pdb file, this assumes that there are no hydrogens to begin with
        :param in_pdb_file: in pdb file
        :param out_pdb_file: out pdb file
        :return: none
        """
        args = ["reduce", self.pdb_file]
        with open(reduced_pdb_file, 'w') as f:
            results = subprocess.run(args, stdout=f)

        if results.returncode != 0:
            raise RuntimeError(f"Reduce failed with return code {results.returncode}")
        else:
            return reduced_pdb_file

    def _run_pdb2xyxzrn(self, reduced_pdb_file="tmp_reduced.pdb", out_xyzr_file="tmp.xyzr"):
        """
        run pdb2xyzrn to convert pdb to xyzrn
        :param in_pdb_file: in pdb file probably from run_reduce
        :param out_xyzrn_file: out xyzrn file
        :return: none
        """
        args = ["pdb_to_xyzr", reduced_pdb_file]
        with open(out_xyzr_file, 'w') as f:
            results = subprocess.run(args, stdout=f)

        if results.returncode != 0:
            raise RuntimeError(f"pdb_to_xyzr failed with return code {results.returncode}")
        else:
            return out_xyzr_file

    def _run_msms(self, in_xyzr_file):
        """
        run msms to compute the molecular surface
        :param in_xyzrn_file: in xyzrn file probably from run_pdb2xyxzrn
        :return: none
        """
        out_base = in_xyzr_file.replace(".xyzrn", "")
        args = ["msms", "-if", in_xyzr_file, "-of", out_base]

        results = subprocess.run(args)
        vert_file = out_base + ".vert"
        face_file = out_base + ".face"
        if results.returncode != 0:
            raise RuntimeError(f"MSMS failed with return code {results.returncode}")
        else:
            return vert_file, face_file

    def _run_pdb2pqr(self, out_pqr_file="tmp.pqr"):
        """

        :param in_pdb_file:
        :return:
        """
        args = ["pdb2pqr", "--ff=PARSE", self.pdb_file, out_pqr_file]
        results = subprocess.run(args)
        if results.returncode != 0:
            raise RuntimeError(f"pdb2pqr failed with return code {results.returncode}")
        else:
            return out_pqr_file

    def _extract_coordinates_from_pdb(self):
        """

        """
        coords = []

        with open(self.pdb_file, 'r') as f:
            for line in f:
                # Parse ATOM and HETATM records
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

        return np.array(coords)

    def _calculate_apbs_grid_bounds(self, padding=10.0):
        """

        :param padding:
        :return:
        """
        # Load coordinates from either PDB or CSV
        coords=self._extract_coordinates_from_pdb()

        # Calculate min/max
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        # Calculate range
        coord_range = max_coords - min_coords

        # Calculate grid lengths with padding
        grid_lengths = coord_range + 2 * padding

        # Calculate center
        center = (min_coords + max_coords) / 2

        return {
            'center': tuple(center),
            'lengths': tuple(grid_lengths),
            'min_bounds': tuple(min_coords),
            'max_bounds': tuple(max_coords),
            'coord_range': tuple(coord_range)
        }

    def _generate_apbs_input(self, output_file="tmp_apbs.in", coordinates_csv=None,
                            padding=10.0, dime=65, ion_conc=0.150):
        """

        :param output_file:
        :param coordinates_csv:
        :param padding:
        :param dime:
        :param ion_conc:
        :return:
        """
        grid_params = self._calculate_apbs_grid_bounds(padding=padding)

        # Ensure dime is odd
        if dime % 2 == 0:
            dime += 1

        # Generate APBS input
        apbs_input = f"""read
        mol pqr tmp.pqr
    end

    elec name solvation
        mg-manual
        dime {dime} {dime} {dime}
        glen {grid_params['lengths'][0]:.3f} {grid_params['lengths'][1]:.3f} {grid_params['lengths'][2]:.3f}
        gcent {grid_params['center'][0]:.3f} {grid_params['center'][1]:.3f} {grid_params['center'][2]:.3f}
        mol 1
        lpbe
        bcfl sdh
        ion charge 1 conc {ion_conc} radius 2.0
        ion charge -1 conc {ion_conc} radius 1.8
        pdie 2.0
        sdie 78.54
        srfm smol
        chgm spl2
        sdens 10.0
        srad 1.4
        swin 0.3
        temp 298.15
        calcenergy total
        calcforce no
        write pot dx tmp_pot
    end

    quit
    """

        with open(output_file, 'w') as f:
            f.write(apbs_input)

    def _run_apbs(self, apbs_outfile="tmp_pot.dx"):
        """
        generate apbs input file
        :param in_pqr_file: in pqr file probably from run_pdb2pqr
        :return: none
        """
        self._generate_apbs_input(output_file="apbs_input.in")

        args=[self.abps_bin, "apbs_input.in"]
        resutls=subprocess.run(args)
        if resutls.returncode != 0:
            raise RuntimeError(f"APBS failed with return code {resutls.returncode}")
        else:
            return apbs_outfile

    def _process_vert_file(self, vert_file, coords_file="coordinates.csv"):
        """
        process vert file to get vertices
        :param vert_file: in vert file probably from run_msms
        :return: vertices as numpy array
        """
        with open(vert_file, "r") as v:
            with open(coords_file, "w") as coords:
                for index, line in enumerate(v):
                    if index < 3:
                        continue
                    else:
                        parts = line.split()
                        x = parts[0]
                        y = parts[1]
                        z = parts[2]
                        coords.write(f"{x},{y},{z}\n")
        # File is now properly closed and flushed
        return coords_file

    def _run_multivalue(self, in_dx_file, in_coords_file, out_csv_file="potentials.csv"):
        """
        run multivalue to get potential values at surface vertices
        :param in_dx_file: in dx file probably from run_apbs
        :param in_coords_file: in coords file probably from process_vert_file
        :return: none
        """
        args = [self.multivalue_bin, in_coords_file, in_dx_file, out_csv_file]
        results = subprocess.run(args)
        if results.returncode != 0:
            raise RuntimeError(f"multivalue failed with return code {results.returncode}")
        else:
            return out_csv_file

    def _cleanup(self, reduced_pdb, xyzr_file, vert_file, face_file, pqr_file, apbs_dx_file, coords_file, potentials_file):
        files=[reduced_pdb, xyzr_file, vert_file, face_file, pqr_file, apbs_dx_file, coords_file, potentials_file]
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def process(self, cleanup=True):
        try:
            reduced_pdb = self._run_reduce()
            xyzrn_file = self._run_pdb2xyxzrn(reduced_pdb)
            vert_file, face_file= self._run_msms(xyzrn_file)
            self._run_pdb2pqr()
            apbs_dx_file = self._run_apbs()
            coords_file = self._process_vert_file(vert_file)
            potentials_file = self._run_multivalue(apbs_dx_file, coords_file)

            #MSMS results processing
            vert_data = np.loadtxt(vert_file, skiprows=3)
            vertices = vert_data[:, 0:3]  # x y z
            normals = vert_data[:, 3:6]  # nx ny nz
            atom_ids = vert_data[:, 7].astype(int) - 1  # closest atom index (1-based to 0-based)

            # .face has 3 headers
            face_data = np.loadtxt(face_file, skiprows=3)
            faces = face_data[:, 0:3].astype(int) - 1  # 1-based to 0-based

            # Potentials processing
            potentials = np.loadtxt(potentials_file, delimiter=',', skiprows=0)
            potential_values = potentials[:, 3]  # assuming fourth column has the values

            traj=md.load(reduced_pdb)
            #TODO, this is not correct
            if cleanup:
                self._cleanup()


            return vertices, faces, normals, atom_ids, potential_values, traj

        except Exception as e:
            print(f"Error processing PDB file: {e}")
            raise e

