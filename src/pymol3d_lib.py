"""
Helper functions for PDB loading, chain/ligand extraction, sequence parsing,
hydrogen-bond detection, PLIP 2D interaction plot, and molecule lookup (PubChem, ChEMBL).
Used by the pymol3d Streamlit app and aligned with pymol3d.ipynb.
"""
import io
import json
import math
import os
import re
from typing import Any, List, Optional, Tuple
from urllib.parse import quote

import requests

# Protein H-bond donor/acceptor heavy-atom names (for distance-based H-bond detection)
PROTEIN_DONORS = {"ND1", "ND2", "NE", "NE2", "NH1", "NH2", "NZ", "OG", "OG1", "OH", "N"}
PROTEIN_ACCEPTORS = {"OD1", "OD2", "OE1", "OE2", "O", "ND1", "ND2", "NE2", "SD"}

# Common solvent/small ions to exclude from "ligand" list
SOLVENT_AND_IONS = {
    "HOH", "WAT", "H2O", "SO4", "GOL", "EDO", "ACT", "PO4", "NA", "CL", "MG",
    "NAG", "NAP", "CA", "ZN", "MN", "FE", "K", "IOD", "BME", "DMS", "FMT",
}

# Three-letter to one-letter amino acid code
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "UNK": "X", "MSE": "M",
}


def fetch_pdb(pdb_id: str) -> str:
    """Fetch PDB file content from RCSB. Raises on error."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def get_chains(pdb_text: str) -> list[str]:
    """Return sorted list of chain IDs present in ATOM/HETATM lines."""
    chains = set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            ch = (line[21:22] or " ").strip() or " "
            chains.add(ch)
    return sorted(chains)


def get_ligands(pdb_text: str) -> list[tuple[str, str]]:
    """Return list of (chain, resname) for non-solvent/ion HETATM groups."""
    ligands = set()
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            ch = (line[21:22] or " ").strip() or " "
            res = (line[17:20] or "").strip()
            if res and res not in SOLVENT_AND_IONS:
                ligands.add((ch, res))
    return sorted(ligands)


def get_ligand_pdb_block(pdb_text: str, ligand_resn: str, chain_id: str) -> str:
    """Extract PDB lines for the given ligand (HETATM + CONECT) as a single block."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM"):
            ch = (line[21:22] or " ").strip() or " "
            res = (line[17:20] or "").strip()
            if ch == chain_id and res == ligand_resn:
                lines.append(line)
        elif line.startswith("CONECT") and lines:
            # Include CONECT that reference atoms we kept (simplified: include all CONECT)
            lines.append(line)
    return "\n".join(lines) if lines else ""


def get_pharmacophore_points(
    ligand_pdb_block: str,
    feature_set: str = "BaseFeatures",
) -> List[Tuple[float, float, float, str]]:
    """
    Get 3D pharmacophore feature points for a ligand using RDKit.

    feature_set: "BaseFeatures" (default) uses RDDataDir/BaseFeatures.fdef;
                 "Gobbi" uses the feature factory from rdkit.Chem.Pharm2D.Gobbi_Pharm2D
                 (Gobbi & Poppinger 2D pharmacophore feature definitions, applied in 3D).
    Returns list of (x, y, z, family). Returns [] if RDKit is unavailable or parse fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import ChemicalFeatures
        from rdkit import RDConfig
        import os
    except ImportError:
        return []
    if not ligand_pdb_block or not ligand_pdb_block.strip():
        return []
    try:
        mol = Chem.MolFromPDBBlock(ligand_pdb_block)
        if mol is None:
            return []
        if mol.GetNumConformers() == 0:
            return []
        conf_id = 0
        factory = None
        if feature_set == "Gobbi":
            try:
                from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
                factory = Gobbi_Pharm2D.factory.featFactory  # MolChemicalFeatureFactory
            except Exception:
                factory = None
        if factory is None:
            fdef_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            if not os.path.isfile(fdef_path):
                return []
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
        feats = factory.GetFeaturesForMol(mol, confId=conf_id)
        out: List[Tuple[float, float, float, str]] = []
        for feat in feats:
            pos = feat.GetPos(conf_id)
            family = feat.GetFamily()
            out.append((float(pos.x), float(pos.y), float(pos.z), family))
        return out
    except Exception:
        return []


def _parse_atoms(
    pdb_text: str,
    chain_id: str,
    het_only: bool,
    ligand_resn: Optional[str] = None,
) -> list[tuple[str, int, str, float, float, float]]:
    """Parse ATOM or HETATM lines. Returns list of (resname, resnum, atom_name, x, y, z)."""
    atoms = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if het_only and line.startswith("ATOM  "):
            continue
        if not het_only and line.startswith("HETATM"):
            continue
        ch = (line[21:22] or " ").strip() or " "
        if ch != chain_id:
            continue
        resname = (line[17:20] or "").strip()
        if ligand_resn is not None and resname != ligand_resn:
            continue
        resnum = int((line[22:26] or "0").strip())
        atom_name = (line[12:16] or "").strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        atoms.append((resname, resnum, atom_name, x, y, z))
    return atoms


def _distance(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def get_binding_site_residues(
    pdb_text: str,
    ligand_resn: str,
    ligand_chain_id: str,
    distance_cutoff: float = 5.0,
) -> List[str]:
    """
    Return list of 'chain:resi' for protein residues that have at least one atom
    within distance_cutoff Å of any ligand atom. Ligand is identified by ligand_resn
    and ligand_chain_id. Excludes solvent/ions.
    """
    ligand_coords: List[Tuple[float, float, float]] = []
    # (chain, resi) -> list of (x,y,z)
    protein_by_res: dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}

    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        ch = (line[21:22] or " ").strip() or " "
        resname = (line[17:20] or "").strip()
        resi = (line[22:26] or "0").strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        pt = (x, y, z)

        if line.startswith("HETATM") and ch == ligand_chain_id and resname == ligand_resn:
            ligand_coords.append(pt)
            continue
        if line.startswith("ATOM  "):
            if resname in SOLVENT_AND_IONS:
                continue
            key = (ch, resi)
            if key not in protein_by_res:
                protein_by_res[key] = []
            protein_by_res[key].append(pt)

    if not ligand_coords:
        return []

    out: List[str] = []
    for (ch, resi), coords in protein_by_res.items():
        min_d = min(_distance(p, lig) for p in coords for lig in ligand_coords)
        if min_d <= distance_cutoff:
            out.append(f"{ch}:{resi}")
    return sorted(out)


def get_residue_labels(
    pdb_text: str,
    res_ids: List[str],
) -> List[Tuple[str, str, str]]:
    """
    For each 'chain:resi' in res_ids, get the residue name (3-letter) from the PDB
    and return a list of (label_str, chain, resi) e.g. ("Lys142", "A", "142").
    Label format: resname + resi (e.g. Lys142). Uses first ATOM line for each residue.
    """
    # (chain, resi) -> resname
    resname_by_key: dict[Tuple[str, str], str] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  "):
            continue
        ch = (line[21:22] or " ").strip() or " "
        resname = (line[17:20] or "").strip()
        resi = (line[22:26] or "0").strip()
        if resname and resname not in SOLVENT_AND_IONS:
            key = (ch, resi)
            if key not in resname_by_key:
                resname_by_key[key] = resname
    out: List[Tuple[str, str, str]] = []
    for res_key in res_ids:
        parts = res_key.split(":", 1)
        if len(parts) != 2:
            continue
        ch, resi = parts[0].strip(), parts[1].strip()
        resname = resname_by_key.get((ch, resi), "UNK")
        # Format: Lys142 (capitalize first letter of 3-letter code for display)
        label = (resname[0].upper() + resname[1:].lower() if len(resname) >= 2 else resname) + resi
        out.append((label, ch, resi))
    return out


def find_hbonds(
    pdb_text: str,
    chain_id: str,
    ligand_resn: str,
    cutoff: float = 3.5,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    Find protein–ligand H-bonds (heavy-atom distance ≤ cutoff).
    Returns list of ((x1,y1,z1), (x2,y2,z2)) for each pair (protein atom, ligand atom).
    """
    protein_atoms = _parse_atoms(pdb_text, chain_id, het_only=False)
    ligand_atoms = _parse_atoms(pdb_text, chain_id, het_only=True, ligand_resn=ligand_resn)
    if not ligand_atoms:
        return []

    protein_do_acc = [
        (r, n, (x, y, z))
        for r, rn, n, x, y, z in protein_atoms
        if n in PROTEIN_DONORS or n in PROTEIN_ACCEPTORS
    ]
    ligand_do_acc = [
        (r, n, (x, y, z))
        for r, rn, n, x, y, z in ligand_atoms
        if n[0] in ("N", "O")
    ]
    pairs = []
    for _pr, _pn, pc in protein_do_acc:
        for _lr, _ln, lc in ligand_do_acc:
            if _distance(pc, lc) <= cutoff:
                pairs.append((pc, lc))
    return pairs


def get_pdb_for_chain(
    pdb_text: str,
    chain: Optional[str] = None,
    include_ligands: bool = True,
) -> str:
    """
    Return PDB content for one chain (and its ligands if include_ligands),
    or full structure if chain is None.
    """
    if chain is None or chain == "All":
        return pdb_text
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            ch = (line[21:22] or " ").strip() or " "
            if ch == chain:
                lines.append(line)
        elif line.startswith("TER") and lines and lines[-1].startswith(("ATOM  ", "HETATM")):
            lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def get_sequence(pdb_text: str, chain: Optional[str] = None) -> list[tuple[str, int, str, str]]:
    """
    Return list of (chain_id, resseq, resname_3, resname_1) for each residue
    (one entry per residue using CA for standard residues, else first atom).
    """
    seen = set()  # (chain, resseq)
    out = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        ch = (line[21:22] or " ").strip() or " "
        if chain is not None and chain != "All" and ch != chain:
            continue
        resseq_s = (line[22:26] or "").strip()
        try:
            resseq = int(re.sub(r"[A-Za-z]", "", resseq_s) or "0")
        except ValueError:
            resseq = 0
        resname = (line[17:20] or "").strip()
        key = (ch, resseq)
        if key in seen:
            continue
        atname = (line[12:16] or "").strip()
        # For ATOM: only count CA so we get one letter per amino acid
        if line.startswith("ATOM  ") and atname != "CA":
            continue
        seen.add(key)
        one = AA3_TO_1.get(resname, "X")
        out.append((ch, resseq, resname, one))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def sequence_string(pdb_text: str, chain: Optional[str] = None) -> str:
    """One-letter sequence for the given chain (or first chain if All)."""
    seq_list = get_sequence(pdb_text, chain)
    return "".join(r[3] for r in seq_list)


def get_residue_list_for_selector(pdb_text: str, chain: Optional[str] = None) -> list[tuple[str, str]]:
    """List of (label, resi_key) for dropdown: label like '62 - ARG (A)', resi_key like 'A:62'."""
    seq = get_sequence(pdb_text, chain)
    return [(f"{r[1]} - {r[2]} ({r[0]})", f"{r[0]}:{r[1]}") for r in seq]


# --- PLIP 2D interaction plot (Protein-Ligand Interaction Profiler) ---

def generate_plip_2d_plot(
    pdb_block: str,
    ligand_resn: str,
    chain_id: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[Optional[Any], Optional[bytes], Optional[str]]:
    """
    Run PLIP on a PDB block and generate a 2D interaction summary plot.

    Uses PLIP to detect H-bonds, hydrophobic, pi-stacking, salt bridges, etc.
    Returns a Plotly figure (if available) and PNG bytes for display/download.

    Parameters
    ----------
    width, height : int, optional
        Plot dimensions in pixels. If None, dimensions are derived from content.

    Returns
    -------
    (plotly_fig, png_bytes, error_message)
        On success: (fig or None, png_bytes, None). On failure: (None, None, str).
    """
    try:
        from plip.structure.preparation import PDBComplex
        import numpy as np
    except ImportError:
        return None, None, (
            "PLIP 2D plot requires plip. PLIP also needs OpenBabel: "
            "install OpenBabel first (e.g. conda install -c conda-forge openbabel), "
            "then pip install openbabel plip. See https://github.com/pharmai/plip"
        )

    try:
        mol = PDBComplex()
        mol.load_pdb(pdb_block, as_string=True)
        if not mol.ligands:
            return None, None, "No ligands found in the structure."
        for ligand in mol.ligands:
            mol.characterize_complex(ligand)
        if not mol.interaction_sets:
            return None, None, "No interactions detected for any ligand."

        # Match binding site: prefer ligand with same resname (and chain if given)
        bsid = None
        for lig in mol.ligands:
            hetid = getattr(lig, "hetid", None) or getattr(lig, "name", "")
            ch = getattr(lig, "chain", "").strip() or ""
            if hetid == ligand_resn and (chain_id is None or ch == chain_id):
                bsid = lig.mol.title
                break
        if bsid is None:
            bsid = next(iter(mol.interaction_sets.keys()))
        inter = mol.interaction_sets.get(bsid)
        if inter is None:
            return None, None, "No interaction set for the selected ligand."

        # PLIP uses separate lists for donor/acceptor etc.; collect (residue_label, interaction_type)
        # Attribute names as in plip.structure.preparation (characterize_complex)
        type_sources = [
            ("Hydrogen bond", ["hbonds_ldon", "hbonds_pdon"]),
            ("Hydrophobic", ["hydrophobic_contacts"]),
            ("Pi-stacking", ["pistacking"]),
            ("Pi-cation", ["pication_laro", "pication_paro"]),
            ("Salt bridge", ["saltbridge_lneg", "saltbridge_pneg"]),
            ("Halogen bond", ["halogen_bonds"]),
            ("Water bridge", ["water_bridges"]),
            ("Metal", ["metal_complexes"]),
        ]
        pairs: List[Tuple[str, str]] = []
        for label, attrs in type_sources:
            for attr in attrs:
                lst = getattr(inter, attr, None)
                if lst is None:
                    continue
                for item in lst:
                    resnr = getattr(item, "resnr", None)
                    restype = getattr(item, "restype", None) or ""
                    reschain = getattr(item, "reschain", None)
                    if reschain is None:
                        reschain = ""
                    reschain = str(reschain).strip()
                    if restype in ("LIG", "HOH", "") or resnr is None:
                        continue
                    res_label = f"{restype}{resnr}.{reschain}"
                    pairs.append((res_label, label))
        if not pairs:
            return None, None, "No interactions detected for the selected ligand."

        # Build grid: residue (y) vs interaction type (x)
        type_order = [t for t, _ in type_sources]
        residues = sorted(set(p[0] for p in pairs))
        types = sorted(set(p[1] for p in pairs), key=lambda x: type_order.index(x) if x in type_order else 99)
        data = set(pairs)
        grid = np.zeros((len(residues), len(types)))
        for i, res in enumerate(residues):
            for j, itype in enumerate(types):
                if (res, itype) in data:
                    grid[i, j] = 1

        # Distinct color per interaction type (bond type)
        type_colors = {
            "Hydrogen bond": "#2563eb",
            "Hydrophobic": "#ea580c",
            "Pi-stacking": "#7c3aed",
            "Pi-cation": "#db2777",
            "Salt bridge": "#dc2626",
            "Halogen bond": "#0891b2",
            "Water bridge": "#0ea5e9",
            "Metal": "#059669",
        }
        # One color per bond type (for types present in this plot)
        plot_colors = [type_colors.get(t, "#64748b") for t in types]
        n_types = len(types)
        absent_color = "#f1f5f9"

        title = f"PLIP interactions — ligand {ligand_resn}" + (f" (chain {chain_id})" if chain_id else "")
        plotly_fig = None
        png_bytes = None

        # z: 0 = absent, 1..n = type index (so each bond type gets its own color)
        z_typed = np.zeros_like(grid, dtype=float)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1:
                    z_typed[i, j] = j + 1

        # Prefer Plotly for interactive, polished plot
        try:
            import plotly.graph_objects as go
            plot_w = width if width is not None else max(500, min(1000, 70 * len(types) + 80))
            plot_h = height if height is not None else max(450, min(1000, 42 * len(residues) + 140))
            hovertemplate = "Residue %{y}<br>Interaction: %{x}<extra></extra>"
            # Discrete colorscale: 0 = absent, 1..n = one color per interaction type
            colorscale = [[0.0, absent_color]]
            for j, c in enumerate(plot_colors):
                colorscale.append([(j + 1) / (n_types + 1), c])
            if n_types > 0:
                colorscale.append([1.0, plot_colors[-1]])
            fig = go.Figure(
                data=go.Heatmap(
                    z=z_typed,
                    x=types,
                    y=residues,
                    colorscale=colorscale,
                    zmin=0,
                    zmax=n_types + 1,
                    showscale=True,
                    hoverongaps=False,
                    hovertemplate=hovertemplate,
                )
            )
            fig.update_layout(
                title=dict(text=title, font=dict(size=16, family="Inter, system-ui, sans-serif")),
                font=dict(family="Inter, system-ui, sans-serif", size=12),
                xaxis=dict(tickangle=-40, tickfont=dict(color="black", size=14, family="Inter, system-ui, sans-serif", weight="bold"), side="bottom"),
                yaxis=dict(autorange="reversed", tickfont=dict(color="black", size=14, family="Inter, system-ui, sans-serif", weight="bold")),
                margin=dict(l=80, r=40, t=60, b=100),
                paper_bgcolor="white",
                plot_bgcolor="white",
                height=plot_h,
                width=plot_w,
            )
            tickvals = [0] + list(range(1, n_types + 1))
            ticktext = ["No"] + types
            fig.update_traces(
                colorbar=dict(title="Bond type", tickvals=tickvals, ticktext=ticktext, len=0.5, tickfont=dict(size=10)),
            )
            buf = io.BytesIO()
            try:
                fig.write_image(buf, format="png", width=plot_w, height=plot_h)
            except Exception:
                buf = None
            if buf:
                buf.seek(0)
                png_bytes = buf.read()
            plotly_fig = fig
        except Exception:
            pass

        # Fallback: matplotlib with improved styling (one color per bond type)
        if png_bytes is None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            if width is not None and height is not None:
                fig_w_in = max(4, width / 100)
                fig_h_in = max(4, height / 100)
            else:
                fig_w_in = max(7, len(types) * 1.4)
                fig_h_in = max(7, len(residues) * 0.55)
            fig, ax = plt.subplots(
                figsize=(fig_w_in, fig_h_in),
                facecolor="white",
            )
            ax.set_facecolor("#fafafa")
            # RGBA image: each cell colored by bond type or absent
            rgb = np.zeros((len(residues), len(types), 4))
            rgba_absent = np.array(mcolors.to_rgba(absent_color))
            for j, c in enumerate(plot_colors):
                rgba = np.array(mcolors.to_rgba(c))
                for i in range(len(residues)):
                    rgb[i, j, :] = rgba if grid[i, j] == 1 else rgba_absent
            im = ax.imshow(rgb, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(residues)))
            ax.set_yticklabels(residues, fontsize=14, fontweight="bold", color="black", fontfamily="sans-serif")
            ax.set_xticks(range(len(types)))
            ax.set_xticklabels(types, rotation=40, ha="right", fontsize=14, fontweight="bold", color="black", fontfamily="sans-serif")
            ax.set_xlabel("Interaction type", fontsize=12)
            ax.set_ylabel("Protein residue", fontsize=12)
            ax.set_title(title, fontsize=14, pad=12)
            ax.tick_params(axis="both", which="major", labelsize=14, labelcolor="black")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#e2e8f0")
            plt.tight_layout()
            buf = io.BytesIO()
            dpi = 150
            if width is not None and height is not None and fig_w_in > 0 and fig_h_in > 0:
                dpi = min(width / fig_w_in, height / fig_h_in)
            plt.savefig(buf, format="png", dpi=max(72, min(300, dpi)), bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)
            png_bytes = buf.read()

        return plotly_fig, png_bytes, None
    except Exception as e:
        return None, None, str(e)


# --- Molecule lookup (PubChem, ChEMBL) for 2D viewer ---

def _lookup_pubchem(name: str) -> Optional[dict[str, Any]]:
    """Look up molecule by name in PubChem. Returns dict with SMILES and properties or None."""
    if not name or not name.strip():
        return None
    name = name.strip()
    props = "CanonicalSMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight,IUPACName,Title"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/{props}/JSON"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        plist = data.get("PropertyTable", {}).get("Properties", [])
        if not plist:
            return None
        p = plist[0]
        smiles = p.get("CanonicalSMILES") or p.get("ConnectivitySMILES")
        if not smiles:
            return None
        return {
            "smiles": smiles,
            "title": p.get("Title") or name,
            "formula": p.get("MolecularFormula"),
            "molecular_weight": p.get("MolecularWeight"),
            "iupac_name": p.get("IUPACName"),
            "cid": p.get("CID"),
            "source": "PubChem",
        }
    except Exception:
        return None


def _lookup_chembl(name: str) -> Optional[dict[str, Any]]:
    """Look up molecule by synonym in ChEMBL. Returns dict with SMILES and properties or None."""
    if not name or not name.strip():
        return None
    name = name.strip()
    try:
        search_url = (
            "https://www.ebi.ac.uk/chembl/api/data/molecule_synonym/search"
            f"?synonym__icontains={quote(name)}&format=json"
        )
        r = requests.get(search_url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        syns = data.get("molecule_synonyms", [])
        if not syns:
            return None
        mol_id = syns[0].get("molecule_chembl_id")
        if not mol_id:
            return None
        mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{mol_id}.json"
        r2 = requests.get(mol_url, timeout=15)
        if r2.status_code != 200:
            return None
        mol = r2.json()
        struct = mol.get("molecule_structures") or {}
        smiles = struct.get("canonical_smiles") if isinstance(struct, dict) else None
        if not smiles:
            return None
        props = mol.get("molecule_properties") or {}
        if not isinstance(props, dict):
            props = {}
        return {
            "smiles": smiles,
            "title": mol.get("pref_name") or name,
            "formula": props.get("molecular_formula"),
            "molecular_weight": props.get("molecular_weight"),
            "iupac_name": props.get("full_molformula") or mol.get("pref_name"),
            "chembl_id": mol_id,
            "source": "ChEMBL",
        }
    except Exception:
        return None


def lookup_molecule_by_name(name: str) -> Optional[dict[str, Any]]:
    """
    Look up a molecule by name from PubChem or ChEMBL.
    Returns a dict with keys: smiles, title, formula, molecular_weight, iupac_name, source (and cid or chembl_id).
    Returns None if not found.
    """
    if not name or not name.strip():
        return None
    result = _lookup_pubchem(name)
    if result:
        return result
    result = _lookup_chembl(name)
    if result:
        return result
    return None


# -----------------------------------------------------------------------------
# OpenBio API (https://openbio.tech) — protein structure tools
# Set OPENBIO_API_KEY in the environment. Base URL: https://api.openbio.tech/api/v1
# -----------------------------------------------------------------------------

OPENBIO_BASE = "https://api.openbio.tech/api/v1"


def openbio_get_api_key() -> Optional[str]:
    """Return OpenBio API key from Streamlit secrets or env OPENBIO_API_KEY."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            # Try [openbio] api_key then top-level OPENBIO_API_KEY
            try:
                ob = st.secrets.get("openbio")
                if isinstance(ob, dict) and ob.get("api_key"):
                    return str(ob["api_key"]).strip()
            except Exception:
                pass
            try:
                k = st.secrets.get("OPENBIO_API_KEY")
                if k:
                    return str(k).strip()
            except Exception:
                pass
            try:
                k = st.secrets["OPENBIO_API_KEY"]
                if k:
                    return str(k).strip()
            except Exception:
                pass
    except Exception:
        pass
    return os.environ.get("OPENBIO_API_KEY") or None


def _openbio_key() -> Optional[str]:
    """Internal: key for API calls (secrets or env)."""
    return openbio_get_api_key()


def _openbio_sanitize_params(params: dict) -> dict:
    """Remove empty strings and None so the API doesn't reject the request."""
    if not isinstance(params, dict):
        return {}
    return {
        k: v for k, v in params.items()
        if v is not None and v != ""
    }


def openbio_invoke(tool_name: str, params: dict) -> Tuple[bool, Any]:
    """
    Invoke an OpenBio tool. Returns (success, result).
    Tries form-encoded body first; if API returns 400/415, tries JSON body.
    """
    key = _openbio_key()
    if not key:
        return False, "OPENBIO_API_KEY is not set. Set it in .streamlit/secrets.toml or environment."
    params = _openbio_sanitize_params(params if isinstance(params, dict) else {})
    headers = {"X-API-Key": key.strip()}
    url = f"{OPENBIO_BASE}/tools"

    def _err(r) -> str:
        try:
            body = r.json()
            if isinstance(body, dict):
                msg = body.get("error") or body.get("message") or body.get("detail")
                if isinstance(msg, str):
                    return msg
                if isinstance(msg, list):
                    return "; ".join(str(m) for m in msg[:5])
        except Exception:
            pass
        return r.text[:600] if r.text else f"HTTP {r.status_code}"

    try:
        # Try 1: form-encoded (tool_name, params as JSON string)
        r = requests.post(
            url,
            headers=headers,
            data={
                "tool_name": tool_name,
                "params": json.dumps(params),
            },
            timeout=60,
        )
        if r.status_code == 200:
            out = r.json()
            if isinstance(out, dict) and out.get("error"):
                return False, out.get("error") or str(out)
            return True, out
        err1 = _err(r)
        # Try 2: JSON body (some APIs expect this)
        if r.status_code in (400, 415, 422):
            r2 = requests.post(
                url,
                headers={**headers, "Content-Type": "application/json"},
                json={"tool_name": tool_name, "params": params},
                timeout=60,
            )
            if r2.status_code == 200:
                out = r2.json()
                if isinstance(out, dict) and out.get("error"):
                    return False, out.get("error") or str(out)
                return True, out
            err1 = _err(r2)
        return False, f"API error {r.status_code}: {err1}"
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def openbio_list_tools() -> Tuple[bool, Any]:
    """List all available OpenBio tools. Returns (success, list_of_tool_names_or_error)."""
    key = _openbio_key()
    if not key:
        return False, "OPENBIO_API_KEY is not set."
    try:
        r = requests.get(
            f"{OPENBIO_BASE}/tools",
            headers={"X-API-Key": key.strip()},
            timeout=30,
        )
        if r.status_code != 200:
            return False, f"API error {r.status_code}: {r.text[:500]}"
        data = r.json()
        if isinstance(data, list):
            return True, data
        if isinstance(data, dict) and "tools" in data:
            return True, data["tools"]
        if isinstance(data, dict) and "result" in data:
            return True, data["result"] if isinstance(data["result"], list) else [data["result"]]
        if isinstance(data, dict):
            # Sometimes API returns {"tool_name": schema, ...}
            keys = [k for k in data.keys() if not k.startswith("_") and isinstance(k, str)]
            if keys:
                return True, keys
        return True, data if isinstance(data, list) else [data]
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def openbio_search_tools(query: str) -> Tuple[bool, Any]:
    """Search OpenBio tools by capability. Returns (success, list_or_error)."""
    key = _openbio_key()
    if not key:
        return False, "OPENBIO_API_KEY is not set."
    try:
        r = requests.get(
            f"{OPENBIO_BASE}/tools/search",
            params={"q": query},
            headers={"X-API-Key": key.strip()},
            timeout=15,
        )
        if r.status_code != 200:
            return False, f"API error {r.status_code}: {r.text[:500]}"
        data = r.json()
        if isinstance(data, list):
            return True, data
        if isinstance(data, dict) and "tools" in data:
            return True, data["tools"]
        if isinstance(data, dict) and "result" in data:
            return True, data["result"] if isinstance(data["result"], list) else [data["result"]]
        return True, data if isinstance(data, list) else [data]
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def openbio_get_tool_schema(tool_name: str) -> Tuple[bool, Any]:
    """Get parameter schema for an OpenBio tool. Returns (success, schema_or_error)."""
    key = _openbio_key()
    if not key:
        return False, "OPENBIO_API_KEY is not set."
    try:
        r = requests.get(
            f"{OPENBIO_BASE}/tools/{tool_name}",
            headers={"X-API-Key": key.strip()},
            timeout=15,
        )
        if r.status_code != 200:
            return False, f"API error {r.status_code}: {r.text[:500]}"
        return True, r.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)
