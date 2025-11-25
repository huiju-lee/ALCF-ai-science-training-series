
from typing import Any, Dict, Literal
import os

from langchain_core.tools import tool
from ase import Atoms
from ase.io import write as ase_write, read as ase_read

@tool
def name_to_smiles_and_xyz(
        name: str,
        output_file: str = "molecule_from_name.xyz",
        random_seed: int = 2025,
        ) -> Dict[str, Any]:
        """
        Given a common molecule name (e.g. 'ethanol'), fetch its SMILES from PubChem,
        build a 3D structure with RDKit, and write an XYZ file.

        Returns a small JSON-like dict with metadata that other tools or agents can use.
        """ 
        import pubchempy
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # 1) name -> SMILES (PubChem)
        compounds = pubchempy.get_compounds(str(name), "name")
        if not compounds:
            raise ValueError(f"Could not find molecule '{name}' in PubChem.")

        smiles = compounds[0].canonical_smiles

        # 2) SMILES -> 3D structure (RDKit)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("PubChem returned an invalid SMILES string.")

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
            raise ValueError("Failed to generate 3D coordinates with RDKit.")
        if AllChem.UFFOptimizeMolecule(mol) != 0:
            raise ValueError("Failed to optimize geometry with RDKit UFF.")

        conf = mol.GetConformer()
        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

        atoms = Atoms(numbers=numbers, positions=positions)
        ase_write(output_file, atoms)

        return {
                "ok": True,
                "artifact": "coordinate_file",
                "type": "xyz",
                "molecule_name": name,
                "smiles": smiles,
                "natoms": len(numbers),
                "path": os.path.abspath(output_file),
                }

@tool
def mace_single_point_energy(
        input_file: str,
        device: Literal["cpu", "cuda"] = "cpu",
        ) -> Dict[str, Any]:
    """
    Run a MACE single-point energy calculation on a structure file.

    This does NOT change the geometry; it just computes the potential energy.
    """
    from mace.calculators import mace_mp

    if not os.path.isfile(input_file):
        raise ValueError(f"Input structure file '{input_file}' does not exist.")

    dev = device.lower()
    if dev not in ("cpu", "cuda"):
        dev = "cpu"

    try:
        atoms = ase_read(input_file)
    except Exception as e:
        raise ValueError(f"Could not read '{input_file}' with ASE: {e}")

    try:
        calc = mace_mp(model=mace_model_name, device=dev)
    except Exception as e:
        raise ValueError(
                f"Could not load MACE model '{mace_model_name}'. Original error: {e}"
                )

    atoms.calc = calc
    energy = float(atoms.get_potential_energy())

    return {
            "status": "success",
            "mode": "single_point",
            "input_file": os.path.abspath(input_file),
            "mace_model_name": mace_model_name,
            "device": dev,
            "energy_eV": energy,
            }

@tool
def mace_geometry_optimization(
        input_file: str,
        output_file: str = "optimized.xyz",
        mace_model_name: str = "small",
        device: Literal["cpu", "cuda"] = "cpu",
        fmax: float = 0.05,
        max_steps: int = 200,
        ) -> Dict[str, Any]:
    """
    Run a geometry optimization with MACE using ASE's BFGS optimizer.
    Writes the optimized structure to an XYZ file and returns useful metadata.
    """
    from mace.calculators import mace_mp
    from ase.optimize import BFGS

    if not os.path.isfile(input_file):
        raise ValueError(f"Input structure file '{input_file}' does not exist.")

    dev = device.lower()
    if dev not in ("cpu", "cuda"):
        dev = "cpu"

    try:
        atoms = ase_read(input_file)
    except Exception as e:
        raise ValueError(f"Could not read '{input_file}' with ASE: {e}")

    try:
        calc = mace_mp(model=mace_model_name, device=dev)
    except Exception as e:
        raise ValueError(
                f"Could not load MACE model '{mace_model_name}'. Original error: {e}"
                )

    atoms.calc = calc

    try:
        opt = BFGS(atoms)
        opt.run(fmax=fmax, steps=max_steps)
        converged = True
    except Exception as e:
        converged = False
        raise ValueError(f"Geometry optimization failed: {e}")

    final_energy = float(atoms.get_potential_energy())
    ase_write(output_file, atoms)

    return {
            "status": "success",
            "mode": "geometry_optimization",
            "converged": converged,
            "input_file": os.path.abspath(input_file),
            "output_file": os.path.abspath(output_file),
            "mace_model_name": mace_model_name,
            "device": dev,
            "final_energy_eV": final_energy,
            "final_positions": atoms.get_positions().tolist(),
            "final_cell": atoms.get_cell().tolist(),
            "fmax_used": fmax,
            "max_steps_used": max_steps,
            }
