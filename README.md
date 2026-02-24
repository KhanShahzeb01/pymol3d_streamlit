# pymol3d — 3D molecular visualization (Streamlit app)

This is a learning / practice project I built to get hands‑on with **Streamlit** and **py3Dmol** for interactive molecular visualization, with coding assistance from **Cursor** (AI pair‑programmer).  
The app supports **protein**, **protein–ligand**, **protein–protein**, and **ligand** visualization using **py3Dmol** (3Dmol.js) in the browser — no PyMOL desktop installation required.

## Layout

- **Center (~65%)**: Interactive 3D viewer (rotate, zoom, pan with mouse).
- **Right bar (~35%)**: PDB load (ID + chain or file upload), chain/ligand selection, protein color & style, ligand color, protein opacity.
- **Center tabs:** 3D viewer, Hydrogen bonds, **2D interaction plot (PLIP)** — [PLIP](https://github.com/pharmai/plip) residue–interaction heatmap and PNG export (optional).

---

## How to use

### 3D viewer mode

1. **Load a structure**  
   In the right panel under **Structure**, enter a **PDB ID** (e.g. `8R4V`) or upload a `.pdb` / `.ent` file, then click **Load structure**.

2. **Choose chain and ligand**  
   Use **Chain** and **Ligand** in the **Chain & ligand** section to focus on a specific chain and ligand (or leave as “All” / “None”).

3. **Adjust the view**  
   In **Representation** you can:
   - Pick a **Preset** (e.g. PyMOL Publication, Ligand cartoon, Chimera Cartoon) or **Custom** for manual control.
   - Set **Protein color** (e.g. by secondary structure, by chain, spectrum, solid).
   - Set **Ligand color** (element/CPK or theme colors).
   - Set **Protein style** (cartoon, surface, lines, sticks, trace) and **Protein opacity**.
   - Change the **3D viewer background** (gray, white, black).

4. **Analysis and tools**  
   In **Analysis** you can enable H-bonds, binding-site focus, and 3D pharmacophore. Under **Tools** use **Residues to focus** and **Protein sequence**. In the viewer area, tabs give **3D viewer**, **Hydrogen bonds**, and **2D interaction plot (PLIP)**.

5. **Viewer** — Rotate, zoom, pan with the mouse; use full-screen in the viewer, **Escape** to exit.

### 2D viewer mode

Switch **Mode** to **2D viewer**. In the right panel use **2D molecule** to look up by name or paste **SMILES**, then **Draw** or **Load into editor**. Use **Molecular descriptors** and **Fingerprints** for properties and batch CSV.

---

## Run the app

From the project root:

```bash
pip install -r requirements.txt
streamlit run src/pymol3d_app.py
```

Optional: `./scripts/run_pymol3d_chrome.sh` (opens in Chrome). Notebook: `notebooks/pymol3d.ipynb`.

---

## Tips

- **3D:** Use **Cartoon** + **By secondary structure** for a classic protein view. Use **Surface** with lower **opacity** to see the binding pocket around the ligand.
- **2D:** ESOL **LogS** is log10 of molar aqueous solubility. Use download buttons to save PNGs and CSV tables.

---

## Acknowledgments

This project was developed with coding assistance from **[Cursor](https://cursor.com)** (AI pair-programmer). Cursor is credited here in the documentation; all commits to this repository are authored by the maintainer.
