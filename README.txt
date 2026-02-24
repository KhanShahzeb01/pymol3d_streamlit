PyMOL3D – Protein & Ligand Viewer
=================================

Overview
--------

This Streamlit app provides:
- 3D visualization of proteins and protein–ligand complexes using py3Dmol.
- A 2D/3D small-molecule editor driven by SMILES.
- Single-molecule and batch molecular descriptor calculations, including QED and ESOL LogS.

Repository layout
-----------------

- pymol3d_app.py  – main Streamlit app entry point (run this file with Streamlit).
- pymol3d_lib.py  – helper functions for PDB loading, chains/ligands, sequences, binding sites, PLIP plots, etc.
- requirements.txt – Python dependencies for the app.

Environment setup
-----------------

1. Create and activate a virtual environment (recommended):

   - On Linux/macOS:
     python -m venv venv
     source venv/bin/activate

   - On Windows (PowerShell):
     python -m venv venv
     venv\Scripts\Activate.ps1

2. Install dependencies:

   pip install --upgrade pip
   pip install -r requirements.txt

3. (Optional) PLIP / OpenBabel for 2D interaction plots:

   - The PLIP 2D interaction plot requires PLIP and OpenBabel.
   - A common way is to install OpenBabel via conda, then PLIP via pip, for example:
     conda install -c conda-forge openbabel
     pip install plip

Running the app
---------------

From the `pymol3d` directory:

   streamlit run pymol3d_app.py

Then open the URL shown in the terminal (typically http://localhost:8501 or similar).

Basic usage
-----------

3D viewer mode:
- Use the right-hand controls to:
  - Load a structure from a PDB ID or upload a local .pdb file.
  - Select a chain and ligand.
  - Adjust protein and ligand style, color, and opacity.
  - Show hydrogen bonds, binding-site residues, and (if enabled) 3D pharmacophore features.

2D viewer mode:
- Use the 2D molecule expander to:
  - Look up molecules by name from PubChem/ChEMBL, or
  - Paste/enter a SMILES string directly.
- Use the 2D editor + 3D window to sketch and visualize small molecules.
- Use the Molecular descriptors section to:
  - Compute single-molecule descriptors (including QED and ESOL LogS), or
  - Upload a CSV with a SMILES column and compute QED + ESOL descriptors in batch, then download the results.

In-app help
-----------

The app includes a **“Help & how to use”** panel in the Streamlit sidebar.  
Open it at any time to see a concise guide to the main features and workflows.

