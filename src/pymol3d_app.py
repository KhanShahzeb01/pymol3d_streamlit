"""
Streamlit app for 3D visualization of proteins, protein-ligand, protein-protein,
and ligands using py3Dmol. Layout: left bar (sequence, image options, 2D molecule),
center viewer, right bar (PDB/upload, chain, ligand, colors, style, opacity).
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, List, Optional
from urllib.parse import quote, unquote

import streamlit as st
import py3Dmol
import pandas as pd

from pymol3d_lib import (
    fetch_pdb,
    find_hbonds,
    generate_plip_2d_plot,
    get_binding_site_residues,
    get_chains,
    get_ligands,
    get_ligand_pdb_block,
    get_pharmacophore_points,
    get_pdb_for_chain,
    get_residue_labels,
    get_sequence,
    get_residue_list_for_selector,
    sequence_string,
    lookup_molecule_by_name,
)

# # Try RDKit for 2D drawing, molecular descriptors, and SMILES→3D
# try:
#     from rdkit import Chem
#     from rdkit.Chem import Draw, AllChem, RDKFingerprint
#     from rdkit.Chem import Descriptors, Crippen
#     from rdkit.Chem import QED
#     try:
#         from rdkit.Chem.MACCSkeys import GenMACCSKeys
#     except Exception:
#         GenMACCSKeys = None
#     HAS_RDKIT = True
# except ImportError:
#     HAS_RDKIT = False
#     AllChem = None
#     RDKFingerprint = None
#     GenMACCSKeys = None

# Try RDKit for 2D drawing, molecular descriptors, and SMILES→3D
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, RDKFingerprint
    from rdkit.Chem import Descriptors, Crippen
    from rdkit.Chem import QED
    try:
        from rdkit.Chem.MACCSkeys import GenMACCSKeys
    except Exception:
        GenMACCSKeys = None
    HAS_RDKIT = True
except Exception as e:  # broaden beyond ImportError
    import streamlit as st
    HAS_RDKIT = False
    AllChem = None
    RDKFingerprint = None
    GenMACCSKeys = None
    st.sidebar.error(f"RDKit import failed: {e!r}")

# Fingerprint types: (display label, key for _compute_fingerprint_single)
FINGERPRINT_OPTIONS = [
    ("RDKit (topological)", "rdkit"),
    ("Morgan ECFP2 (radius=2, 2048 bits)", "morgan2"),
    ("Morgan ECFP4 (radius=2, 2048 bits)", "morgan4"),
    ("MACCS keys (166 bits)", "maccs"),
    ("Atom pair (hashed, 2048 bits)", "atompair"),
    ("Topological torsion (hashed, 2048 bits)", "torsion"),
]

# Curated RDKit descriptor names for the 2D molecule properties table (subset of Chem.Descriptors.CalcMolDescriptors)
# QED is computed separately via Chem.QED.qed(mol)
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "ExactMolWt", "HeavyAtomMolWt", "HeavyAtomCount",
    "MolLogP", "MolMR", "TPSA", "LabuteASA",
    "NumHDonors", "NumHAcceptors", "NHOHCount", "NOCount",
    "NumRotatableBonds", "NumHeteroatoms", "RingCount", "FractionCSP3",
    "BalabanJ", "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "Ipc", "Chi0", "Chi0n", "Chi1", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
    "MaxAbsEStateIndex", "MinAbsEStateIndex", "MaxPartialCharge", "MinPartialCharge",
    "QED",  # Quantitative Estimate of Drug-likeness (rdkit.Chem.QED)
    # ESOL aqueous solubility model (log10 molar solubility) and component properties
    "ESOL_LogS",
    "ESOL_MolLogP",
    "ESOL_MolWt",
    "ESOL_NumRotBonds",
    "ESOL_AromaticProportion",
]

# Help text shown in the sidebar Help panel
HELP_MARKDOWN = """
### How to use PyMOL3D

- **Modes**  
  - **3D viewer**: Load a protein structure (PDB ID or file upload), choose chain and ligand, and explore in 3D.  
  - **2D viewer**: Sketch or paste a small-molecule SMILES, preview 2D/3D, and compute properties.

- **3D viewer workflow**  
  1. In the right panel, go to **Structure** and either enter a **PDB ID** or upload a `.pdb` file.  
  2. Click **Load structure**.  
  3. Use the **Chain** and **Ligand** selectors to focus on a specific chain and ligand.  
  4. Adjust **style, colors, opacity, and background** using the controls on the right.  
  5. Use the **Tools** section to highlight residues, view the protein sequence, show H‑bonds, binding-site residues, and 3D pharmacophore points.  
  6. In the 3D viewer tabs you can see the main 3D view, H‑bond summary, and the PLIP 2D interaction plot (if PLIP is installed).

- **2D viewer workflow**  
  1. Switch to **2D viewer** mode (top of the app).  
  2. In the **2D molecule** expander, either:  
     - Look up a molecule by name (PubChem / ChEMBL) **or**  
     - Paste a **SMILES** string directly.  
  3. Click **Draw** to preview a 2D image, or **Load into editor** to send it to the main 2D/3D editor.  
  4. In **Molecular descriptors → Single molecule**, select descriptors (including **QED** and **ESOL_LogS**) and click **Calculate property**.  
  5. In **Molecular descriptors → Batch (CSV)**, upload a CSV with a `SMILES` column to compute **QED + ESOL LogS and component properties** for many molecules and download the results.

- **Tips**  
  - ESOL **LogS** is log10 of molar aqueous solubility (mol/L).  
  - Use the download buttons to save **PNG images** and **CSV tables** for further analysis.  
  - If something looks wrong, check the console / terminal for error messages (missing packages, PLIP/OpenBabel, etc.).
"""

# Layout: viewer on left (or top row), single tools sidebar on right
VIEWER_COL_WEIGHT = 4  # ~67%
TOOLS_COL_WEIGHT = 2   # ~33%, min-width enforced via CSS

# 3Dmol.js: use "color" for spectrum and solid; use "colorscheme" for ssPyMol, ssJmol, chain
PROTEIN_COLOR_OPTIONS = {
    "By secondary structure (PyMOL)": "ssPyMol",
    "By secondary structure (Jmol)": "ssJmol",
    "By spectrum (N→C)": "spectrum",
    "By chain": "chain",
    "Solid (light grey)": "lightgrey",
    "Solid (white)": "white",
    "Solid (light blue)": "lightblue",
}

# Ligand: Element (CPK) / schemes, or theme color (backbone in that color, other atoms by element)
LIGAND_COLOR_OPTIONS = {
    "Element (CPK)": "element",
    "Default": "default",
    "Chain": "chain",
    "Spectrum": "spectrum",
}

# 3D display panel background options (label -> hex for 3Dmol and CSS)
VIEWER_BACKGROUND_OPTIONS = [
    ("Gray", "0xeeeeee", "#eeeeee"),
    ("White", "0xffffff", "#ffffff"),
    ("Black", "0x000000", "#000000"),
]

# 2D viewer: 3D panel representation style (label -> value passed to JS)
REPRESENTATION_2D_OPTIONS = [
    ("Sticks", "stick"),
    ("Lines", "line"),
    ("Licorice", "licorice"),
]

# Theme colors: backbone (C/H) uses this; O, N, S, etc. keep element colors (see ELEMENT_COLORS_CPK)
LIGAND_THEME_COLORS = [
    "orange", "red", "blue", "green", "yellow", "magenta", "cyan", "purple",
    "pink", "lime", "salmon", "teal", "darkorange", "hotpink",
]

# CPK-like colors for non-carbon atoms when a theme color is used (C/H get the theme)
ELEMENT_COLORS_CPK = {
    "O": "red",
    "N": "blue",
    "S": "yellow",
    "P": "orange",
    "F": "green",
    "Cl": "green",
    "Br": "darkorange",
    "I": "purple",
    "H": None,  # use theme color
    "C": None,  # use theme color
}

PROTEIN_STYLE_OPTIONS = {
    "Cartoon": "cartoon",
    "Surface": "surface",
    "Lines": "line",
    "Sticks": "stick",
    "Trace (backbone)": "trace",
}

# Presets modeled on PyMOL, Chimera, Maestro, and Discovery Studio (protein / protein–ligand / protein–protein)
# Tuple: (preset_id, display_label, protein_style, protein_color, protein_opacity, ligand_color, background_hex)
PRESET_CUSTOM = "custom"
VISUALIZATION_PRESETS = [
    (PRESET_CUSTOM, "Custom (manual settings)", None, None, None, None, "0xeeeeee"),
    # PyMOL (preset.publication, preset.ligand_cartoon, preset.simple)
    ("pymol_publication", "PyMOL: Publication", "cartoon", "ssPyMol", 0.9, "element", "0xffffff"),
    ("pymol_ligand_cartoon", "PyMOL: Ligand cartoon", "cartoon", "ssPyMol", 0.85, "element", "0xeeeeee"),
    ("pymol_simple", "PyMOL: Simple", "cartoon", "spectrum", 0.9, "element", "0xeeeeee"),
    ("pymol_ligand_sites", "PyMOL: Ligand sites (surface)", "surface", "ssPyMol", 0.5, "element", "0xeeeeee"),
    # Chimera / ChimeraX (Cartoon preset, Ribbons)
    ("chimera_cartoon", "Chimera: Cartoon", "cartoon", "chain", 0.9, "element", "0xeeeeee"),
    ("chimera_ribbons", "Chimera: Ribbons", "cartoon", "ssJmol", 0.85, "element", "0xdddddd"),
    # Maestro (Style Toolbox: cartoon + CPK ligand)
    ("maestro_ligand", "Maestro: Protein–ligand", "cartoon", "ssPyMol", 0.9, "element", "0xffffff"),
    ("maestro_clean", "Maestro: Clean", "cartoon", "lightgrey", 0.85, "element", "0xffffff"),
    # Discovery Studio (ribbon, cartoon, chain colors)
    ("ds_ribbon", "Discovery Studio: Ribbon", "cartoon", "chain", 0.9, "element", "0xf5f5f5"),
    ("ds_cartoon", "Discovery Studio: Cartoon (ss)", "cartoon", "ssPyMol", 0.88, "element", "0xeeeeee"),
    ("ds_surface", "Discovery Studio: Surface", "surface", "chain", 0.55, "element", "0xeeeeee"),
]

# Pharmacophore feature family -> 3Dmol color (BaseFeatures + Gobbi_Pharm2D family names)
PHARMACOPHORE_COLORS = {
    "Donor": "blue",
    "Acceptor": "red",
    "Aromatic": "green",
    "Hydrophobe": "yellow",
    "LumpedHydrophobe": "yellow",
    "PosIonizable": "blue",
    "NegIonizable": "red",
    "ZnBinder": "gray",
    # Gobbi_Pharm2D families (HD=Donor, HA=Acceptor, AR=Aromatic, etc.)
    "HD": "blue",
    "HA": "red",
    "AR": "green",
    "LH": "yellow",
    "RR": "gray",
    "X": "gray",
    "BG": "blue",
    "AG": "red",
}


def _make_viewer_html(
    pdb_block: str,
    protein_style: str,
    protein_color: str,
    protein_opacity: float,
    ligand_resn: Optional[str],
    ligand_color: str,
    focus_res_ids: Optional[list[str]],
    width: int,
    height: int,
    hbond_pairs: Optional[list[tuple[tuple[float, float, float], tuple[float, float, float]]]] = None,
    binding_site_res_ids: Optional[list[str]] = None,
    hide_distant_protein: bool = False,
    binding_site_style: str = "stick",
    background_color: str = "0xeeeeee",
    pharmacophore_points: Optional[list[tuple[float, float, float, str]]] = None,
    pharmacophore_sphere_radius: float = 0.5,
    binding_site_labels: Optional[list[tuple[str, str, str]]] = None,
    label_font_size: int = 14,
) -> str:
    """Build py3Dmol viewer and return HTML string. focus_res_ids: list of 'chain:resi' to highlight as sticks.
    pharmacophore_points: list of (x, y, z, family) to draw as colored spheres (RDKit pharmacophore).
    pharmacophore_sphere_radius: radius (Å) of pharmacophore spheres.
    hbond_pairs: list of ((x1,y1,z1), (x2,y2,z2)) to draw as dashed yellow cylinders.
    binding_site_res_ids: when set, focus view on ligand + these residues (stick/line); hide_distant_protein grays out the rest.
    background_color: hex string for viewer background (e.g. 0xffffff).
    binding_site_labels: when set, list of (label_str, chain, resi) e.g. ('Lys142','A','142') to add residue labels.
    label_font_size: font size in px for residue labels. Label text color follows display background (white on black, black on white/gray)."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_block, "pdb")
    view.setBackgroundColor(background_color)

    use_binding_site = binding_site_res_ids and len(binding_site_res_ids) > 0
    if use_binding_site and hide_distant_protein:
        # Distant protein: very faint so binding site stands out
        faint_spec = {"line": {"opacity": 0.15, "color": "lightgrey"}}
        view.setStyle(faint_spec)
    else:
        # Protein style and color (3Dmol: spectrum uses "color"; ssPyMol/ssJmol/chain use "colorscheme")
        use_colorscheme = protein_color in ("ssPyMol", "ssJmol", "chain")
        if protein_style == "surface":
            # 3Dmol.js requires addSurface() for surfaces; setStyle with surface does not create one
            surface_style = {"opacity": max(0.5, protein_opacity)}
            if use_colorscheme:
                surface_style["colorscheme"] = protein_color
            else:
                surface_style["color"] = protein_color
            view.addSurface(py3Dmol.SAS, surface_style, {})
        elif protein_style == "cartoon":
            style_spec = {"cartoon": {"opacity": protein_opacity}}
            if use_colorscheme:
                style_spec["cartoon"]["colorscheme"] = protein_color
            else:
                style_spec["cartoon"]["color"] = protein_color
            view.setStyle(style_spec)
        else:
            style_spec = {protein_style: {"opacity": protein_opacity}}
            if use_colorscheme:
                style_spec[protein_style]["colorscheme"] = protein_color
            else:
                style_spec[protein_style]["color"] = protein_color
            view.setStyle(style_spec)

    # Binding site residues: stick or line (override protein style)
    bs_style_key = "stick" if binding_site_style == "stick" else "line"
    bs_style = (
        {"stick": {"radius": 0.25, "colorscheme": "element"}}
        if binding_site_style == "stick"
        else {"line": {"colorscheme": "element"}}
    )
    if use_binding_site:
        for res_key in binding_site_res_ids:
            parts = res_key.split(":", 1)
            if len(parts) == 2:
                view.setStyle({"chain": parts[0], "resi": parts[1]}, bs_style)

    # Ligand: theme color (backbone C/H in theme, O/N/S etc. in element colors) or colorscheme
    if ligand_resn:
        base_stick = {"radius": 0.2}
        if ligand_color in ("element", "default", "chain", "spectrum"):
            base_stick["colorscheme"] = ligand_color
            view.setStyle({"resn": ligand_resn}, {"stick": base_stick})
        elif ligand_color in LIGAND_THEME_COLORS:
            # Backbone (C, H and rest) in theme color; O, N, S, etc. overridden with element colors
            theme = ligand_color
            view.setStyle({"resn": ligand_resn}, {"stick": {**base_stick, "color": theme}})
            for elem, color in ELEMENT_COLORS_CPK.items():
                if color is not None:
                    view.setStyle({"resn": ligand_resn, "elem": elem}, {"stick": {**base_stick, "color": color}})
        else:
            view.setStyle({"resn": ligand_resn}, {"stick": {**base_stick, "colorscheme": "element"}})

    # Focus residues (manual selection): sticks and zoom (only if not using binding-site focus)
    focus_list = focus_res_ids if focus_res_ids else []
    if not use_binding_site:
        stick_style = {"stick": {"radius": 0.25, "colorscheme": "element"}}
        for res_key in focus_list:
            parts = res_key.split(":", 1)
            if len(parts) == 2:
                chain_id, resi = parts[0], parts[1]
                view.setStyle({"chain": chain_id, "resi": resi}, stick_style)

    # Zoom: binding site (ligand + residues) or focus list or ligand or all
    if use_binding_site and ligand_resn:
        # Zoom to ligand + binding site residues (3Dmol accepts "or" selection)
        or_sel = [{"resn": ligand_resn}]
        for res_key in binding_site_res_ids:
            parts = res_key.split(":", 1)
            if len(parts) == 2:
                or_sel.append({"chain": parts[0], "resi": parts[1]})
        try:
            view.zoomTo({"or": or_sel})
        except Exception:
            view.zoomTo({"resn": ligand_resn})
    elif focus_list:
        if len(focus_list) == 1:
            first = focus_list[0].split(":", 1)
            if len(first) == 2:
                view.zoomTo({"chain": first[0], "resi": first[1]})
        else:
            view.zoomTo()
    elif ligand_resn:
        view.zoomTo({"resn": ligand_resn})
    else:
        view.zoomTo()

    # Hydrogen bonds: dashed cylinders between protein and ligand atoms
    if hbond_pairs:
        for (x1, y1, z1), (x2, y2, z2) in hbond_pairs:
            view.addCylinder({
                "start": {"x": x1, "y": y1, "z": z1},
                "end": {"x": x2, "y": y2, "z": z2},
                "radius": 0.15,
                "color": "yellow",
                "dashed": True,
            })

    # 3D pharmacophore points (RDKit ChemicalFeatures or Gobbi_Pharm2D) as semi-transparent spheres
    if pharmacophore_points:
        r = max(0.1, float(pharmacophore_sphere_radius))
        for x, y, z, family in pharmacophore_points:
            color = PHARMACOPHORE_COLORS.get(family, "gray")
            view.addSphere({
                "center": {"x": x, "y": y, "z": z},
                "radius": r,
                "color": color,
                "opacity": 0.6,
            })

    # Residue labels for binding site (only those focused by "Focus on binding site")
    # Text color follows display background: white when background is black, black otherwise (no label box)
    # 3Dmol LabelSpec uses fontSize (number) and font (name) separately so size is applied
    if binding_site_labels:
        label_font_color = "white" if (background_color or "").strip().lower() == "0x000000" else "black"
        label_style_spec = {
            "fontColor": label_font_color,
            "fontSize": int(label_font_size),
            "font": "bold Arial",
            "showBackground": False,
        }
        for label_text, chain, resi in binding_site_labels:
            view.addLabel(
                label_text,
                label_style_spec,
                {"chain": chain, "resi": resi},
            )

    raw_html = view._make_html()
    # Wrap in a container with a fullscreen button. When fullscreen, viewer expands to fill screen.
    wrapped = f'''
    <style>
    #pymol3d-viewer-wrap:fullscreen {{
        width: 100% !important;
        height: 100% !important;
        min-width: 100vw !important;
        min-height: 100vh !important;
        display: block !important;
        background: #e0e0e0;
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
    }}
    #pymol3d-viewer-wrap:fullscreen [id^="3dmolviewer"] {{
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100% !important;
        height: 100% !important;
        min-width: 100% !important;
        min-height: 100% !important;
    }}
    #pymol3d-viewer-wrap:fullscreen canvas {{
        width: 100% !important;
        height: 100% !important;
        display: block !important;
    }}
    </style>
    <div id="pymol3d-viewer-wrap" style="position:relative;display:inline-block;">
        <button type="button" id="pymol3d-fs-btn" style="position:absolute;top:8px;right:8px;z-index:9999;padding:6px 10px;
                cursor:pointer;border:1px solid #999;border-radius:4px;background:#888;color:#fff;font-size:12px;">
            ⛶ Full screen
        </button>
        <button type="button" id="pymol3d-reset-btn" style="position:absolute;bottom:8px;right:8px;z-index:9999;padding:6px 10px;
                cursor:pointer;border:1px solid #999;border-radius:4px;background:#888;color:#fff;font-size:12px;">
            Reset view
        </button>
        <button type="button" id="pymol3d-dl-png-btn" style="position:absolute;bottom:8px;right:95px;z-index:9999;padding:6px 10px;
                cursor:pointer;border:1px solid #999;border-radius:4px;background:#888;color:#fff;font-size:12px;">
            Download PNG
        </button>
        {raw_html}
    </div>
    <div id="pymol3d-dl-modal" style="display:none;position:fixed;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:10000;align-items:center;justify-content:center;">
        <div id="pymol3d-dl-modal-box" style="background:#fff;padding:20px;border-radius:8px;min-width:280px;box-shadow:0 4px 20px rgba(0,0,0,0.3);" onclick="event.stopPropagation();">
            <p style="margin:0 0 12px 0;font-weight:bold;">Set image resolution (px)</p>
            <label style="display:block;margin-bottom:4px;">Width</label>
            <input type="number" id="pymol3d-dl-w" value="1000" min="400" max="3840" step="100" style="width:100%;padding:6px;margin-bottom:10px;box-sizing:border-box;">
            <label style="display:block;margin-bottom:4px;">Height</label>
            <input type="number" id="pymol3d-dl-h" value="850" min="300" max="2160" step="100" style="width:100%;padding:6px;margin-bottom:14px;box-sizing:border-box;">
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button type="button" id="pymol3d-dl-cancel" style="padding:6px 14px;cursor:pointer;background:#ddd;border:1px solid #999;border-radius:4px;">Cancel</button>
                <button type="button" id="pymol3d-dl-save" style="padding:6px 14px;cursor:pointer;background:#888;color:#fff;border:1px solid #999;border-radius:4px;">Save PNG</button>
            </div>
        </div>
    </div>
    <script>
    (function() {{
        var wrap = document.getElementById("pymol3d-viewer-wrap");
        var btn = document.getElementById("pymol3d-fs-btn");
        var resetBtn = document.getElementById("pymol3d-reset-btn");
        var dlPngBtn = document.getElementById("pymol3d-dl-png-btn");
        if (!wrap || !btn) return;

        var modal = document.getElementById("pymol3d-dl-modal");
        var dlW = document.getElementById("pymol3d-dl-w");
        var dlH = document.getElementById("pymol3d-dl-h");
        var dlCancel = document.getElementById("pymol3d-dl-cancel");
        var dlSave = document.getElementById("pymol3d-dl-save");

        function downloadViewerPNGAtResolution() {{
            var viewerEl = wrap.querySelector("[id^='3dmolviewer']");
            if (!viewerEl) return;
            var canvas = viewerEl.querySelector("canvas");
            if (!canvas || !canvas.toDataURL) return;
            var w = Math.max(400, Math.min(3840, parseInt(dlW && dlW.value ? dlW.value : 1000, 10) || 1000));
            var h = Math.max(300, Math.min(2160, parseInt(dlH && dlH.value ? dlH.value : 850, 10) || 850));
            try {{
                var off = document.createElement("canvas");
                off.width = w;
                off.height = h;
                var ctx = off.getContext("2d");
                ctx.fillStyle = "#eeeeee";
                ctx.fillRect(0, 0, w, h);
                ctx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, w, h);
                var dataUrl = off.toDataURL("image/png");
                var a = document.createElement("a");
                a.href = dataUrl;
                a.download = "protein_viewer.png";
                a.click();
            }} catch (e) {{}}
            if (modal) modal.style.display = "none";
        }}

        if (dlPngBtn) dlPngBtn.onclick = function() {{ if (modal) modal.style.display = "flex"; }};
        if (dlCancel) dlCancel.onclick = function() {{ if (modal) modal.style.display = "none"; }};
        if (dlSave) dlSave.onclick = downloadViewerPNGAtResolution;
        if (modal) modal.onclick = function(e) {{ if (e.target === modal) modal.style.display = "none"; }};

        function resetView() {{
            function fitWholeProtein(viewer) {{
                if (!viewer || !viewer.zoomTo) return;
                viewer.zoomTo();
                if (viewer.render) viewer.render();
            }}
            var viewerEl = wrap.querySelector("[id^='3dmolviewer']");
            if (viewerEl && viewerEl.id) {{
                var vid = viewerEl.id.replace("3dmolviewer_", "");
                var v = window["viewer_" + vid];
                if (v && typeof v.zoomTo === "function") {{
                    try {{ fitWholeProtein(v); }} catch (e) {{}}
                    return;
                }}
            }}
            if (typeof $3Dmol !== "undefined" && $3Dmol.viewers && $3Dmol.viewers.length) {{
                for (var i = 0; i < $3Dmol.viewers.length; i++) fitWholeProtein($3Dmol.viewers[i]);
            }}
        }}
        if (resetBtn) resetBtn.onclick = resetView;

        function resizeViewer() {{
            var viewerEl = wrap.querySelector("[id^='3dmolviewer']");
            if (!viewerEl) return;
            viewerEl.style.width = "100%";
            viewerEl.style.height = "100%";
            var w = wrap.clientWidth;
            var h = wrap.clientHeight;
            if (viewerEl.style.width !== "100%") viewerEl.style.width = "100%";
            if (viewerEl.style.height !== "100%") viewerEl.style.height = "100%";
            window.dispatchEvent(new Event("resize"));
            var vid = viewerEl.id ? viewerEl.id.replace("3dmolviewer_", "") : "";
            if (vid && typeof window["viewer_" + vid] === "function") {{
                try {{ window["viewer_" + vid]().resize(); }} catch (e) {{}}
            }}
            if (typeof $3Dmol !== "undefined" && $3Dmol.viewers && $3Dmol.viewers.length) {{
                for (var i = 0; i < $3Dmol.viewers.length; i++) {{
                    try {{ if ($3Dmol.viewers[i].resize) $3Dmol.viewers[i].resize(); }} catch (e) {{}}
                }}
            }}
        }}

        function onFullscreenChange() {{
            if (document.fullscreenElement === wrap) {{
                wrap.style.width = "100%";
                wrap.style.height = "100%";
                wrap.style.position = "fixed";
                wrap.style.left = "0";
                wrap.style.top = "0";
                wrap.style.right = "0";
                wrap.style.bottom = "0";
                var viewerEl = wrap.querySelector("[id^='3dmolviewer']");
                if (viewerEl) {{
                    viewerEl.style.position = "absolute";
                    viewerEl.style.left = "0";
                    viewerEl.style.top = "0";
                    viewerEl.style.right = "0";
                    viewerEl.style.bottom = "0";
                    viewerEl.style.width = "100%";
                    viewerEl.style.height = "100%";
                }}
                function doResize() {{
                    var cw = wrap.clientWidth;
                    var ch = wrap.clientHeight;
                    if (viewerEl && cw && ch) {{
                        viewerEl.style.width = cw + "px";
                        viewerEl.style.height = ch + "px";
                    }}
                    resizeViewer();
                }}
                setTimeout(doResize, 0);
                setTimeout(doResize, 100);
                setTimeout(doResize, 300);
            }}
        }}

        document.addEventListener("fullscreenchange", onFullscreenChange);
        document.addEventListener("webkitfullscreenchange", onFullscreenChange);

        btn.onclick = function() {{
            if (!document.fullscreenElement) {{
                if (wrap.requestFullscreen) wrap.requestFullscreen();
                else if (wrap.webkitRequestFullscreen) wrap.webkitRequestFullscreen();
            }} else {{
                if (document.exitFullscreen) document.exitFullscreen();
                else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            }};
        }};
        document.addEventListener("keydown", function(e) {{
            if (e.key === "Escape" && document.fullscreenElement) {{
                if (document.exitFullscreen) document.exitFullscreen();
                else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            }}
        }});
    }})();
    </script>
    '''
    return wrapped


def render_2d_mol(smiles: str, size: tuple[int, int] = (400, 400)) -> Optional[bytes]:
    """Draw 2D molecule with RDKit; return PNG bytes or None."""
    if not HAS_RDKIT or not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _compute_esol_from_mol(mol: Optional["Chem.Mol"]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[float]]:
    """
    ESOL aqueous solubility model (Delaney) for a given RDKit molecule.
    Returns (logS, MolLogP, MolWt, NumRotBonds, aromatic_proportion).
    logS is log10 of molar solubility in water (mol/L).
    """
    if mol is None:
        return None, None, None, None, None
    try:
        n_heavy = mol.GetNumHeavyAtoms()
    except Exception:
        return None, None, None, None, None
    if not n_heavy:
        return None, None, None, None, None
    try:
        logp = float(Crippen.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
        rb = int(Descriptors.NumRotatableBonds(mol))
        aromatic_count = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        ap = float(aromatic_count) / float(n_heavy) if n_heavy else 0.0
        log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
        return round(float(log_s), 3), logp, mw, rb, ap
    except Exception:
        return None, None, None, None, None


def smiles_to_3d_mol_block(smiles: str) -> Optional[str]:
    """Convert SMILES to 3D MOL block using RDKit (AddHs, Embed, MMFF). Return None on failure."""
    if not HAS_RDKIT or AllChem is None or not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            return None
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=200) != 0:
            pass  # still use the embedded coords
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def _make_3d_molecule_viewer_html(mol_block: str, width: int = 500, height: int = 400) -> str:
    """Build py3Dmol HTML for a single small molecule (MOL block). Stick style, element colors."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(mol_block, "sdf")
    view.setStyle({"stick": {"colorscheme": "element"}})
    view.setBackgroundColor("0xeeeeee")
    view.zoomTo()
    raw_html = view._make_html()
    return raw_html


def _make_combined_editor_3d_html(
    initial_smiles: str = "",
    mol_block: Optional[str] = None,
    total_width: int = 900,
    total_height: int = 580,
    background_color_2d: str = "#eeeeee",
    representation_2d: str = "stick",
    thickness_2d: float = 0.2,
) -> str:
    """Build a single HTML document: left (2D JSME editor) and right (3D viewer).
    Integrates RDKit-JS for direct browser-side 3D coordinate generation and sync.
    When mol_block is provided (server-side 3D), it is injected so the 3D panel displays it immediately.
    background_color_2d: CSS hex (e.g. #eeeeee) for the 3D panel background.
    representation_2d: 'stick' | 'line' | 'licorice' for the 3D molecule style.
    thickness_2d: stick radius / sphere scale (e.g. 0.1–0.5) for 3D display."""
    raw = (initial_smiles or "").strip()
    safe_smiles_js = raw.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
    # Escape mol_block for embedding in a JavaScript string (backslashes, quotes, newlines)
    if mol_block and mol_block.strip():
        safe_mol_block_js = (
            mol_block.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )
        initial_mol_block_js = f"'{safe_mol_block_js}'"
    else:
        initial_mol_block_js = "null"

    # RDKit-JS and py3Dmol CDN links
    RDKIT_JS_URL = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js"
    JSME_URL = "https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"
    
    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>2D/3D Molecular Workbench</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <script>window.initialMolBlock = {initial_mol_block_js};</script>
  <script src="{RDKIT_JS_URL}"></script>
  <style>
    :root {{
      --primary: #4f46e5;
      --primary-hover: #4338ca;
      --bg: #f8fafc;
      --panel-bg: #ffffff;
      --text: #1e293b;
      --border: #e2e8f0;
      --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ 
      margin: 0; padding: 0; height: 100%; width: 100%; 
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      overflow: hidden;
    }}
    .container {{
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100vw;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.5rem 1rem;
      background: var(--panel-bg);
      border-bottom: 1px solid var(--border);
      gap: 1rem;
      flex-shrink: 0;
      box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }}
    .toolbar-group {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .btn {{
      padding: 0.4rem 0.8rem;
      border-radius: 0.375rem;
      border: 1px solid var(--border);
      background: var(--panel-bg);
      color: var(--text);
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{
      background: #f1f5f9;
      border-color: #cbd5e1;
    }}
    .btn-primary {{
      background: var(--primary);
      color: white;
      border: none;
    }}
    .btn-primary:hover {{
      background: var(--primary-hover);
    }}
    .workspace {{
      display: flex;
      flex: 1;
      min-height: 0;
    }}
    .pane {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
      position: relative;
    }}
    .pane-2d {{
      border-right: 1px solid var(--border);
      background: white;
    }}
    .pane-3d {{
      background: {background_color_2d};
    }}
    #jsme_container {{
      flex: 1;
      width: 100%;
    }}
    #viewer_container {{
      flex: 1;
      width: 100%;
      height: 100%;
      min-height: 280px;
    }}
    .status-bar {{
      padding: 0.25rem 1.25rem;
      background: white;
      border-top: 1px solid var(--border);
      font-size: 0.75rem;
      color: #64748b;
      display: flex;
      justify-content: space-between;
    }}
    input[type="text"] {{
      padding: 0.4rem 0.65rem;
      border: 1px solid var(--border);
      border-radius: 0.375rem;
      font-size: 0.875rem;
      width: 250px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="toolbar">
      <div class="toolbar-group">
        <input type="text" id="smilesInput" placeholder="Enter SMILES..." value="">
        <button class="btn" onclick="loadSmilesToEditor()">Load</button>
      </div>
      <div class="toolbar-group">
        <button class="btn btn-primary" id="btnUpdate3D">Update 3D</button>
        <button class="btn" id="btnFullscreen">Fullscreen</button>
        <button class="btn" id="btnExport">Save 3D PNG</button>
      </div>
    </div>
    <div class="workspace">
      <div class="pane pane-2d">
        <div id="jsme_container"></div>
      </div>
      <div class="pane pane-3d" id="pane3d">
        <div id="viewer_container"></div>
      </div>
    </div>
    <div class="status-bar">
      <span id="rdkit-status">Initializing RDKit...</span>
      <span id="molecule-status">No molecule loaded</span>
    </div>
  </div>

  <script src="{JSME_URL}"></script>
  <script>
    var RDKit = null;
    var viewer = null;
    var jsmeApplet = null;
    var repStyle = "{representation_2d}";
    var repThickness = {thickness_2d};

    function getStyleSpec() {{
      var r = repThickness;
      var s = Math.max(0.15, r * 1.2);
      if (repStyle === "line") return {{ line: {{ colorscheme: "element" }} }};
      if (repStyle === "licorice") return {{ stick: {{ colorscheme: "element", radius: r }}, sphere: {{ scale: s }} }};
      return {{ stick: {{ colorscheme: "element", radius: r }}, sphere: {{ scale: s }} }};
    }}

    function render3D(data, format) {{
      if (!viewer) {{
        viewer = $3Dmol.createViewer("viewer_container", {{ backgroundColor: "{background_color_2d}" }});
      }}
      viewer.clear();
      viewer.addModel(data, format);
      viewer.setStyle({{}}, getStyleSpec());
      viewer.zoomTo();
      viewer.render();
      document.getElementById('molecule-status').textContent = '3D Sync OK';
    }}

    // Create 3D viewer and show server-provided molecule immediately (from molecules edit panel)
    if (typeof $3Dmol !== "undefined") {{
      viewer = $3Dmol.createViewer("viewer_container", {{ backgroundColor: "{background_color_2d}" }});
      if (window.initialMolBlock) {{
        try {{
          render3D(window.initialMolBlock, "sdf");
        }} catch (e) {{ console.warn("Initial 3D render:", e); }}
      }}
    }}

    // Initialize RDKit
    initRDKitModule().then(function(instance) {{
      RDKit = instance;
      document.getElementById('rdkit-status').textContent = 'RDKit Ready (WASM)';
      console.log("RDKit Version: " + RDKit.version());
      
      // Auto-sync after RDKit is ready (only if we don't already have a molecule from server)
      if (!window.initialMolBlock) setTimeout(update3D, 1000);
    }});

    function jsmeOnLoad() {{
      jsmeApplet = new JSApplet.JSME("jsme_container", "100%", "100%", {{
        "options" : "query,hydrogens"
      }});
      // Load initial smiles
      var initialSmi = "{safe_smiles_js}";
      if (initialSmi) {{
        jsmeApplet.readGenericMolecularInput(initialSmi);
        document.getElementById('smilesInput').value = initialSmi;
      }}
      
      // Listen for changes (debounced)
      let timeout = null;
      jsmeApplet.setChangeListener(function() {{
        clearTimeout(timeout);
        timeout = setTimeout(update3D, 1500); // 1.5s delay for auto-sync
      }});
    }}

    function loadSmilesToEditor() {{
      var smi = document.getElementById('smilesInput').value.trim();
      if (jsmeApplet && smi) {{
        jsmeApplet.readGenericMolecularInput(smi);
        update3D();
      }}
    }}

    function update3D() {{
      if (!jsmeApplet || !RDKit) return;
      
      const smiles = jsmeApplet.smiles();
      if (!smiles) return;
      
      document.getElementById('smilesInput').value = smiles;
      document.getElementById('molecule-status').textContent = 'Generating 3D: ' + (smiles.length > 30 ? smiles.substring(0, 27) + '...' : smiles);

      try {{
        let mol = RDKit.get_mol(smiles);
        if (mol) {{
          // Generate 3D coordinates in the browser!
          // Minimal lib doesn't have AllChem.EmbedMolecule, 
          // but we can use RDKit.get_mol(smiles).get_new_coords() or similar if available,
          // or we can generate a 3D block if the WASM build supports it.
          // For the standard minimal lib, we might need to fallback to a server request 
          // if complex 3D embedding is needed, BUT for "direct sync", we can at least
          // refresh the view. 
          
          // Actually, let's use the power of 3Dmol.js for a placeholder or fetch from server-side.
          // BUT wait, I can make a hidden XHR to the Streamlit app or use a public API like PubChem for 3D.
          
          fetch(`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${{encodeURIComponent(smiles)}}/record/SDF?record_type=3d`)
            .then(res => res.text())
            .then(sdf => {{
              if (sdf.includes("molecule")) {{
                render3D(sdf, "sdf");
              }} else {{
                // If PubChem 3D fails, use the RDKit-JS 2D coords as a flat 3D model
                const molBlock = mol.get_molblock();
                render3D(molBlock, "mol");
              }}
            }})
            .catch(() => {{
              const molBlock = mol.get_molblock();
              render3D(molBlock, "mol");
            }});
          
          mol.delete();
        }}
      }} catch (e) {{
        console.error("RDKit error:", e);
      }}
    }}

    // Buttons
    document.getElementById('btnUpdate3D').onclick = update3D;
    
    document.getElementById('btnFullscreen').onclick = function() {{
      const elem = document.getElementById('pane3d');
      if (elem.requestFullscreen) elem.requestFullscreen();
      else if (elem.webkitRequestFullscreen) elem.webkitRequestFullscreen();
    }};

    document.getElementById('btnExport').onclick = function() {{
      if (!viewer) return;
      const canvas = document.querySelector('#viewer_container canvas');
      if (canvas) {{
        const link = document.createElement('a');
        link.download = 'molecule_3d.png';
        link.href = canvas.toDataURL("image/png");
        link.click();
      }}
    }};

    // Responsive resize
    window.addEventListener('resize', () => {{
      if (viewer) viewer.resize();
    }});
  </script>
</body>
</html>'''


def _make_molview_editor_html(initial_smiles: str = "", width: int = 900, height: int = 580) -> str:
    """Build HTML that embeds MolView (molview.org) in an iframe. Optionally preload via ?q= search."""
    raw = (initial_smiles or "").strip()
    q = quote(raw) if raw else ""
    src = f"https://molview.org/?q={q}" if q else "https://molview.org/"
    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>MolView – Structure editor</title>
  <style>
    body {{ margin: 0; background: #f5f5f5; font-family: sans-serif; }}
    .molview-wrap {{ padding: 8px; }}
    .molview-wrap iframe {{ border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  </style>
</head>
<body>
  <div class="molview-wrap">
    <iframe src="{src}" width="{width}" height="{height}" allowfullscreen style="display:block;"></iframe>
  </div>
</body>
</html>'''


def _make_jsme_editor_html(initial_smiles: str = "", width: int = 520, height: int = 380) -> str:
    """Build HTML that embeds JSME via programmatic API (JSApplet.JSME). SMILES are drawn in the canvas."""
    raw = (initial_smiles or "").strip()
    safe_smiles = raw.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    safe_smiles_js = raw.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
    w_px = str(width) + "px"
    h_px = str(height) + "px"
    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>JSME Editor</title>
  <style>
    body {{ font-family: sans-serif; margin: 8px; background: #f5f5f5; }}
    .jsme-row {{ display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
    .jsme-box {{ background: #fff; padding: 8px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    label {{ display: block; font-size: 12px; color: #555; margin-bottom: 4px; }}
    input[type="text"] {{ width: 100%; max-width: 400px; padding: 6px; box-sizing: border-box; }}
    button {{ padding: 6px 12px; cursor: pointer; background: #4a90d9; color: #fff; border: none; border-radius: 4px; font-size: 13px; }}
    button:hover {{ background: #357abd; }}
    #smilesOut {{ font-family: monospace; font-size: 12px; }}
    #loadStatus {{ font-size: 12px; color: #666; margin-top: 4px; }}
    #jsme_container {{ min-height: 300px; }}
    .jsme-toolbar-row {{ display: flex; width: 100%; gap: 8px; margin-bottom: 8px; box-sizing: border-box; }}
    .jsme-toolbar-col {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
    .jsme-toolbar-col .jsme-box {{ flex: 1; display: flex; flex-direction: column; }}
    .jsme-toolbar-col input[type="text"] {{ flex: 1; min-height: 2.2em; max-width: none; width: 100%; box-sizing: border-box; }}
    .jsme-btn-row {{ display: flex; flex-direction: row; gap: 8px; margin-top: 6px; flex-wrap: wrap; }}
  </style>
</head>
<body>
  <div class="jsme-toolbar-row">
    <div class="jsme-toolbar-col">
      <div class="jsme-box">
        <label>Load SMILES into editor (draws in canvas below)</label>
        <input type="text" id="smilesIn" value="{safe_smiles}" placeholder="e.g. c1ccccc1 or paste SMILES" />
        <button type="button" id="btnLoad" style="margin-top:6px">Load into editor</button>
        <div id="loadStatus"></div>
      </div>
    </div>
    <div class="jsme-toolbar-col">
      <div class="jsme-box">
        <label>Current SMILES</label>
        <input type="text" id="smilesOut" readonly placeholder="Draw or edit above, then click Get SMILES" />
        <div class="jsme-btn-row">
          <button type="button" id="btnGet">Get SMILES</button>
          <button type="button" id="btnCopy" class="secondary">Copy to clipboard</button>
        </div>
      </div>
    </div>
  </div>
  <div class="jsme-row">
    <div id="jsme_container"></div>
  </div>
  <script>
var jsmeApplet = null;
var initialSmilesForJSME = "{safe_smiles_js}";
function jsmeOnLoad() {{
  try {{
    if (typeof JSApplet === "undefined" || !JSApplet.JSME) {{ document.getElementById("loadStatus").textContent = "JSME not loaded."; return; }}
    jsmeApplet = new JSApplet.JSME("jsme_container", "{w_px}", "{h_px}");
    if (initialSmilesForJSME) {{
      try {{
        jsmeApplet.readGenericMolecularInput(initialSmilesForJSME);
        document.getElementById("loadStatus").textContent = "Molecule loaded from sidebar.";
      }} catch (e) {{ document.getElementById("loadStatus").textContent = ""; }}
    }}
  }} catch (e) {{ document.getElementById("loadStatus").textContent = "Init: " + (e.message || e); }}
}}
  </script>
  <script src="https://jsme-editor.github.io/dist/jsme/jsme.nocache.js"></script>
  <script>
(function() {{
  function setStatus(msg) {{ var el = document.getElementById("loadStatus"); if (el) el.textContent = msg || ""; }}
  function loadSmiles() {{
    var s = document.getElementById("smilesIn").value.trim();
    if (!jsmeApplet) {{ setStatus("Editor not ready. Wait a moment and try again."); return; }}
    try {{
      jsmeApplet.readGenericMolecularInput(s || "");
      setStatus(s ? "Loaded into canvas." : "Cleared.");
    }} catch (e) {{ setStatus("Error: " + (e.message || e)); }}
  }}
  function updateSmilesOut() {{
    var out = document.getElementById("smilesOut");
    if (jsmeApplet && typeof jsmeApplet.smiles === "function") {{ try {{ out.value = jsmeApplet.smiles() || ""; }} catch (e) {{ out.value = ""; }} }}
  }}
  function copyToClipboard() {{
    var out = document.getElementById("smilesOut"); out.select();
    try {{ document.execCommand("copy"); }} catch (e) {{ if (navigator.clipboard) navigator.clipboard.writeText(out.value); }}
  }}
  document.getElementById("btnLoad").onclick = loadSmiles;
  document.getElementById("btnGet").onclick = updateSmilesOut;
  document.getElementById("btnCopy").onclick = function() {{ updateSmilesOut(); copyToClipboard(); }};
}})();
  </script>
</body>
</html>'''


def _table_value_str(v: Any) -> str:
    """Convert a descriptor/value to string for st.table (Arrow-safe; avoids float/object column errors)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else str(v)  # NaN check
    return str(v)


def _fp_bitvec_to_str(fp) -> str:
    """Convert RDKit bit vector to string of 0/1."""
    if fp is None:
        return ""
    n = fp.GetNumBits()
    return "".join("1" if fp.GetBit(i) else "0" for i in range(n))


def _compute_fingerprint_single(mol, fp_key: str):
    """Compute one fingerprint for a molecule. Returns bit vector or None."""
    if mol is None:
        return None
    try:
        if fp_key == "rdkit" and RDKFingerprint is not None:
            return RDKFingerprint(mol)
        if fp_key == "morgan2" and AllChem is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        if fp_key == "morgan4" and AllChem is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)  # ECFP4 radius 3
        if fp_key == "maccs" and GenMACCSKeys is not None:
            return GenMACCSKeys(mol)
        if fp_key == "atompair":
            return Chem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
        if fp_key == "torsion":
            return Chem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
    except Exception:
        return None
    return None


def _compute_fingerprints_for_smiles_list(
    smiles_list: list[str], fp_key: str
) -> list[tuple[str, str, str]]:
    """Parse SMILES, compute fingerprint; return list of (smiles, status, fp_bits_string). status is 'ok' or error message."""
    if not HAS_RDKIT:
        return []
    results = []
    for smi in smiles_list:
        smi = (smi or "").strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append((smi, "Invalid SMILES", ""))
            continue
        fp = _compute_fingerprint_single(mol, fp_key)
        if fp is None:
            results.append((smi, "FP failed", ""))
            continue
        results.append((smi, "ok", _fp_bitvec_to_str(fp)))
    return results


def _fingerprints_to_csv(rows: list[tuple[str, str, str]], fp_type_label: str) -> str:
    """Build CSV: SMILES, Status, Fingerprint (or columns SMILES, Status, fp_type_label)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["SMILES", "Status", f"Fingerprint_{fp_type_label.replace(' ', '_')}"])
    for r in rows:
        w.writerow(list(r))
    return buf.getvalue()


def _molecule_info_and_properties_csv(mol_info: Optional[dict], descriptor_table: dict[str, Any]) -> str:
    """Build a CSV string combining basic molecule info and molecular properties (Section, Property, Value)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Section", "Property", "Value"])
    if mol_info:
        if mol_info.get("title"):
            writer.writerow(["Basic information", "Name", mol_info["title"]])
        if mol_info.get("smiles"):
            writer.writerow(["Basic information", "SMILES", mol_info["smiles"]])
        if mol_info.get("formula"):
            writer.writerow(["Basic information", "Molecular formula", mol_info["formula"]])
        if mol_info.get("molecular_weight") is not None:
            writer.writerow(["Basic information", "Molecular weight", str(mol_info["molecular_weight"])])
        if mol_info.get("iupac_name"):
            writer.writerow(["Basic information", "IUPAC name", mol_info["iupac_name"]])
        if mol_info.get("source"):
            writer.writerow(["Basic information", "Source", mol_info["source"]])
        if mol_info.get("cid") is not None:
            writer.writerow(["Basic information", "PubChem CID", str(mol_info["cid"])])
        if mol_info.get("chembl_id"):
            writer.writerow(["Basic information", "ChEMBL ID", mol_info["chembl_id"]])
    for prop, val in (descriptor_table or {}).items():
        writer.writerow(["Molecular properties", prop, val])
    return buf.getvalue()


def _open_download_2d_dialog() -> None:
    """Open a modal dialog to set resolution and download 2D PNG."""
    if not HAS_RDKIT:
        return
    try:
        dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
        if dialog is None:
            st.warning("Download 2D PNG requires Streamlit 1.33+ (st.dialog). Use the expander below.")
            return
    except AttributeError:
        return

    @dialog("Download 2D PNG – Set resolution")
    def _download_2d_dialog():
        st.caption("Set image resolution (px), then save the 2D structure as PNG.")
        cw, ch = st.columns(2)
        with cw:
            w = st.number_input("Width (px)", min_value=200, max_value=2400, value=800, key="dl_2d_modal_w")
        with ch:
            h = st.number_input("Height (px)", min_value=200, max_value=2400, value=800, key="dl_2d_modal_h")
        smiles = (st.session_state.get("smiles_3d") or st.session_state.get("smiles_2d") or st.session_state.get("smiles") or "").strip()
        if not smiles:
            st.warning("Enter SMILES in the main form first, then open this dialog again.")
            return
        png = render_2d_mol(smiles.strip(), (w, h))
        if png:
            st.download_button("Save PNG", data=png, file_name="molecule_2d.png", mime="image/png", key="dl_2d_modal_btn")

    _download_2d_dialog()


def _open_download_plip_dialog() -> None:
    """Open a modal dialog to set resolution and download PLIP 2D plot as PNG."""
    try:
        dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
        if dialog is None:
            st.warning("Download PLIP PNG requires Streamlit 1.33+ (st.dialog).")
            return
    except AttributeError:
        return

    @dialog("Download PLIP plot PNG – Set resolution")
    def _download_plip_dialog():
        st.caption("Set image resolution (px), then save the PLIP interaction plot as PNG.")
        cw, ch = st.columns(2)
        with cw:
            w = st.number_input("Width (px)", min_value=300, max_value=4000, value=600, key="dl_plip_modal_w")
        with ch:
            h = st.number_input("Height (px)", min_value=300, max_value=4000, value=500, key="dl_plip_modal_h")
        plip_fig = st.session_state.get("plip_2d_fig")
        plip_png = st.session_state.get("plip_2d_png")
        plip_pdb = st.session_state.get("plip_pdb")
        plip_ligand = st.session_state.get("plip_ligand")
        plip_chain = st.session_state.get("plip_chain")
        png_bytes = None
        if plip_fig is not None:
            buf = io.BytesIO()
            try:
                plip_fig.write_image(buf, format="png", width=w, height=h)
                buf.seek(0)
                png_bytes = buf.read()
            except Exception:
                pass
        elif plip_pdb and plip_ligand is not None:
            _, png_bytes, _ = generate_plip_2d_plot(
                plip_pdb, plip_ligand, chain_id=plip_chain, width=w, height=h
            )
        if png_bytes:
            st.download_button("Save PNG", data=png_bytes, file_name="plip_interaction_plot.png", mime="image/png", key="dl_plip_modal_btn")
        else:
            st.warning("Generate the PLIP plot first in the 2D interaction plot tab, then open this dialog again.")

    _download_plip_dialog()


#
# (OpenBio integration removed)
#

# Human-readable names and one-line descriptions for each tool (so the UI is clear).
OPENBIO_TOOL_DESCRIPTIONS = {
    "fetch_pdb_metadata": ("Fetch PDB metadata", "Get resolution, ligands, and binding site info for a PDB ID."),
    "get_alphafold_prediction": ("Get AlphaFold model", "Fetch AlphaFold predicted structure for a UniProt ID."),
    "get_binding_site_residues": ("Binding site residues", "List residues in the binding site of a PDB structure."),
    "get_structures_for_protein": ("Find best structures", "Find PDB structures for a protein (UniProt ID), ranked by resolution/coverage."),
    "search_pdb_text": ("Search PDB by text", "Search structures by keyword, organism, or ligand name."),
    "get_uniprot_entry": ("UniProt entry", "Get protein function, domains, and cross-references from UniProt."),
    "search_pubmed": ("Search PubMed", "Search biomedical literature by query; returns titles and PMIDs."),
    "arxiv_search": ("Search arXiv", "Search physics/CS/math preprints."),
    "biorxiv_search_keywords": ("Search bioRxiv", "Search life sciences preprints by keywords."),
    "fetch_full_text": ("Fetch full text", "Get abstract or full text when available (e.g. PMC)."),
    "openalex_search": ("Search OpenAlex", "Search 240M+ works across all fields."),
    "search_pubmed_author": ("PubMed by author", "Find papers by author name."),
    "lookup_gene": ("Look up gene", "Get gene info by symbol, Ensembl ID, or NCBI Gene ID."),
    "vep_predict": ("VEP: variant effect", "Annotate variants with consequence, SIFT, PolyPhen."),
    "gwas_search_associations_by_trait": ("GWAS by trait", "Search disease/trait associations in GWAS Catalog."),
    "get_gene_sequence": ("Get gene sequence", "Retrieve genomic, transcript, or protein sequence."),
    "convert_gene_ids": ("Convert gene IDs", "Map between Ensembl, UniProt, RefSeq, etc."),
    "geo_search": ("Search GEO", "Search gene expression datasets."),
    "submit_blast": ("Submit BLAST", "Start a BLAST search (protein or nucleotide)."),
    "check_blast_status": ("Check BLAST status", "Poll whether a BLAST job has finished."),
    "get_blast_results": ("Get BLAST results", "Retrieve hits, e-value, identity % for a completed job."),
    "calculate_molecular_properties": ("Molecular properties", "Compute MW, LogP, TPSA, H-bond donors/acceptors from SMILES."),
    "chembl_similarity_search": ("ChEMBL similarity", "Find similar compounds in ChEMBL by SMILES (Tanimoto)."),
    "search_pubchem": ("Search PubChem", "Look up compounds by name or identifier."),
    "validate_smiles": ("Validate SMILES", "Validate and canonicalize a SMILES string."),
    "check_pains": ("Check PAINS", "Detect PAINS/structural alerts from SMILES."),
    "lipinski_rule_of_five": ("Lipinski Rule of 5", "Check drug-likeness (MW, LogP, HBD, HBA)."),
    "design_primers": ("Design primers", "Design PCR primers with Tm, GC%, specificity."),
    "restriction_digest": ("Restriction digest", "Find cut sites and simulate a digest."),
    "assemble_gibson": ("Gibson assembly", "Simulate seamless homology-based assembly."),
    "simulate_gel": ("Simulate gel", "Predict electrophoresis band pattern."),
    "evaluate_primers": ("Evaluate primers", "Check hairpins, dimers, Tm, product size."),
    "simulate_pcr": ("Simulate PCR", "Simulate amplification on a template."),
    "submit_boltz_prediction": ("Submit Boltz prediction", "Start structure prediction (async); use check_job_status next."),
    "submit_proteinmpnn_prediction": ("Submit ProteinMPNN", "Start sequence design job (async)."),
    "submit_chai_prediction": ("Submit Chai prediction", "Start Chai structure prediction (async)."),
    "check_job_status": ("Check job status", "Check if an async prediction job is done."),
    "get_job_result": ("Get job result", "Get download URL and result for a completed job."),
    "analyze_gene_list": ("Analyze gene list", "Pathway or enrichment on a list of genes."),
    "get_string_network": ("STRING network", "Get protein–protein interaction network."),
    "go_enrichment": ("GO enrichment", "Gene Ontology enrichment for a gene list."),
    "kegg_pathway_search": ("KEGG pathway search", "Search KEGG pathways."),
    "reactome_enrichment": ("Reactome enrichment", "Reactome pathway enrichment for gene list."),
    "search_clinical_trials": ("Search ClinicalTrials.gov", "Find trials by condition, intervention, phase."),
    "clinvar_search": ("Search ClinVar", "Variant pathogenicity by gene or position."),
    "get_known_drugs_for_disease": ("Known drugs for disease", "Open Targets: drugs linked to a disease."),
    "fda_adverse_events": ("FDA adverse events", "Drug adverse event reports (FAERS)."),
    "open_targets_search": ("Open Targets search", "Drug–target and disease evidence."),
}

# Fallback: when the API returns no schema, show these inputs so you always have a simple form.
# Each entry: list of (param_name, label, placeholder, type_str).
OPENBIO_KNOWN_PARAMS: dict[str, list[tuple[str, str, str, str]]] = {
    "fetch_pdb_metadata": [("pdb_id", "PDB ID", "e.g. 8R4V", "string")],
    "get_alphafold_prediction": [("uniprot_id", "UniProt ID", "e.g. P12345", "string")],
    "get_binding_site_residues": [("pdb_id", "PDB ID", "e.g. 8R4V", "string"), ("ligand_id", "Ligand ID (optional)", "e.g. STI", "string")],
    "get_structures_for_protein": [("uniprot_id", "UniProt ID", "e.g. P12345", "string")],
    "search_pdb_text": [("query", "Search query", "e.g. kinase", "string")],
    "get_uniprot_entry": [("uniprot_id", "UniProt ID", "e.g. P12345", "string")],
    "search_pubmed": [("query", "Search query", "e.g. cancer kinase", "string"), ("max_results", "Max results", "e.g. 10", "integer")],
    "arxiv_search": [("query", "Search query", "e.g. protein folding", "string"), ("max_results", "Max results", "e.g. 10", "integer")],
    "biorxiv_search_keywords": [("query", "Keywords", "e.g. CRISPR", "string"), ("max_results", "Max results", "e.g. 10", "integer")],
    "fetch_full_text": [("pmid", "PMID or PMC ID", "e.g. 12345678", "string")],
    "openalex_search": [("query", "Search query", "e.g. machine learning biology", "string"), ("max_results", "Max results", "e.g. 10", "integer")],
    "search_pubmed_author": [("author", "Author name", "e.g. Smith J", "string"), ("max_results", "Max results", "e.g. 10", "integer")],
    "lookup_gene": [("gene_id", "Gene symbol or ID", "e.g. BRCA1 or ENSG00000012048", "string")],
    "vep_predict": [("variant", "Variant", "e.g. rs699 or 21 26960070 G A", "string")],
    "gwas_search_associations_by_trait": [("trait", "Trait or disease", "e.g. type 2 diabetes", "string")],
    "get_gene_sequence": [("gene_id", "Gene ID", "e.g. BRCA1", "string"), ("sequence_type", "Sequence type", "genomic / transcript / protein", "string")],
    "convert_gene_ids": [("gene_id", "Gene ID", "e.g. BRCA1", "string"), ("from_db", "From DB", "e.g. ensembl", "string"), ("to_db", "To DB", "e.g. uniprot", "string")],
    "geo_search": [("query", "Search query", "e.g. cancer RNA-seq", "string")],
    "submit_blast": [("sequence", "Sequence (FASTA or raw)", "paste sequence", "string"), ("program", "Program", "blastp / blastn / etc.", "string")],
    "check_blast_status": [("job_id", "BLAST job ID", "from submit_blast", "string")],
    "get_blast_results": [("job_id", "BLAST job ID", "from submit_blast", "string")],
    "calculate_molecular_properties": [("smiles", "SMILES", "e.g. CCO", "string")],
    "chembl_similarity_search": [("smiles", "SMILES", "e.g. CC(=O)Oc1ccccc1C(=O)O", "string"), ("threshold", "Similarity threshold", "e.g. 0.7", "string")],
    "search_pubchem": [("query", "Compound name or ID", "e.g. aspirin", "string")],
    "validate_smiles": [("smiles", "SMILES", "e.g. CCO", "string")],
    "check_pains": [("smiles", "SMILES", "e.g. CCO", "string")],
    "lipinski_rule_of_five": [("smiles", "SMILES", "e.g. CCO", "string")],
    "design_primers": [("sequence", "Template sequence", "DNA sequence", "string"), ("target_start", "Target start", "e.g. 1", "integer"), ("target_end", "Target end", "e.g. 100", "integer")],
    "restriction_digest": [("sequence", "DNA sequence", "paste sequence", "string"), ("enzyme", "Enzyme name(s)", "e.g. EcoRI", "string")],
    "assemble_gibson": [("fragments", "Fragments (JSON array)", "[\"ATCG...\", \"...\"]", "string")],
    "simulate_gel": [("sequences", "Sequences / lengths", "comma-separated or JSON", "string")],
    "evaluate_primers": [("forward", "Forward primer", "e.g. ATGC...", "string"), ("reverse", "Reverse primer", "e.g. GCAT...", "string"), ("template", "Template sequence", "optional", "string")],
    "simulate_pcr": [("template", "Template sequence", "DNA", "string"), ("forward", "Forward primer", "e.g. ATGC...", "string"), ("reverse", "Reverse primer", "e.g. GCAT...", "string")],
    "submit_boltz_prediction": [("sequence", "Protein sequence", "single letter", "string")],
    "submit_proteinmpnn_prediction": [("pdb_id", "PDB ID", "e.g. 8R4V", "string"), ("chain", "Chain", "e.g. A", "string")],
    "submit_chai_prediction": [("sequence", "Protein sequence", "single letter", "string")],
    "check_job_status": [("job_id", "Job ID", "from submit_*", "string")],
    "get_job_result": [("job_id", "Job ID", "from submit_*", "string")],
    "analyze_gene_list": [("gene_list", "Gene list", "comma-separated or one per line", "string"), ("analysis", "Analysis type", "pathway / enrichment", "string")],
    "get_string_network": [("protein_ids", "Protein IDs", "comma-separated", "string")],
    "go_enrichment": [("gene_list", "Gene list", "comma-separated", "string")],
    "kegg_pathway_search": [("query", "Search query", "e.g. glycolysis", "string")],
    "reactome_enrichment": [("gene_list", "Gene list", "comma-separated", "string")],
    "search_clinical_trials": [("condition", "Condition or disease", "e.g. diabetes", "string"), ("max_results", "Max results", "e.g. 20", "integer")],
    "clinvar_search": [("gene", "Gene symbol", "e.g. BRCA1", "string"), ("variant", "Variant (optional)", "e.g. rs699", "string")],
    "get_known_drugs_for_disease": [("disease_id", "Disease ID (EFO or name)", "e.g. EFO_0000405", "string")],
    "fda_adverse_events": [("drug_name", "Drug name", "e.g. aspirin", "string")],
    "open_targets_search": [("query", "Target or disease", "e.g. BRCA1", "string")],
}


def _openbio_tool_label(tool_id: str) -> str:
    """Human-readable label for a tool: 'Display name — Short description'."""
    info = OPENBIO_TOOL_DESCRIPTIONS.get(tool_id)
    if info:
        return f"{info[0]} — {info[1]}"
    return tool_id.replace("_", " ").title()


def _openbio_parse_schema_params(schema: Any) -> List[tuple]:
    """Extract (name, type, required, default, description) from OpenBio tool schema. Tries multiple response shapes."""
    if not isinstance(schema, dict):
        return []
    # Unwrap: API may return { "schema": {...} } or { "data": {...} }
    for key in ("schema", "input_schema", "data"):
        if key in schema and isinstance(schema[key], dict):
            out = _openbio_parse_schema_params(schema[key])
            if out:
                return out
    if "parameters" in schema and isinstance(schema["parameters"], list):
        out = []
        for p in schema["parameters"]:
            if isinstance(p, dict) and p.get("name"):
                sub = p.get("schema") or p
                out.append((
                    p["name"],
                    sub.get("type", "string") if isinstance(sub, dict) else "string",
                    p.get("required", False),
                    p.get("default"),
                    (p.get("description") or (sub.get("description") if isinstance(sub, dict) else "") or ""),
                ))
        return out
    props = schema.get("properties")
    required = schema.get("required") or []
    if not props and "input" in schema and isinstance(schema["input"], dict):
        props = schema["input"].get("properties")
        required = schema["input"].get("required") or []
    if not props or not isinstance(props, dict):
        return []
    return [
        (name, spec.get("type", "string") if isinstance(spec, dict) else "string", name in required, spec.get("default") if isinstance(spec, dict) else None, spec.get("description", "") if isinstance(spec, dict) else "")
        for name, spec in props.items()
    ]


def _openbio_result_to_table(out: Any) -> Optional[List[dict]]:
    """Convert OpenBio result to a list of flat dicts for table display. Returns None if not tabular."""
    if out is None:
        return None
    # Unwrap common wrapper
    if isinstance(out, dict) and "result" in out:
        out = out["result"]
    rows = []
    if isinstance(out, list) and out:
        if isinstance(out[0], dict):
            rows = out
        else:
            rows = [{"value": v} for v in out]
    elif isinstance(out, dict):
        # Dict with a list of dicts inside
        for key in ("results", "hits", "data", "entries", "papers", "items", "records", "list"):
            if isinstance(out.get(key), list) and out[key] and isinstance(out[key][0], dict):
                rows = out[key]
                break
        else:
            rows = [out]
    else:
        return None
    # Flatten values so table cells are simple (no nested dicts/lists)
    def _cell(v: Any) -> Any:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, (dict, list)):
            s = json.dumps(v, default=str)
            return s if len(s) <= 500 else s[:497] + "..."
        return str(v)

    return [{k: _cell(v) for k, v in row.items()} for row in rows]


def _openbio_render_tool_form(tool_name: str, fetch_pdb) -> None:
    """Render one tool: description, then one input per parameter (with label + placeholder), Run button, then Results."""
    info = OPENBIO_TOOL_DESCRIPTIONS.get(tool_name)
    if info:
        st.markdown(f"**What it does:** {info[1]}")
    # Get schema and param list; if API gives nothing, use known params so there are always real input boxes
    ok, schema = openbio_get_tool_schema(tool_name)
    if not ok:
        st.error(schema)
        return
    param_specs = _openbio_parse_schema_params(schema)
    known = OPENBIO_KNOWN_PARAMS.get(tool_name)
    if not param_specs and known:
        param_specs = [(n, "integer" if t == "integer" else t, False, None, "") for n, _lab, _ph, t in known]
    known_map = {n: (lab, ph) for n, lab, ph, _ in (known or [])}
    st.markdown("**Inputs** — fill these and click **Run** below.")
    params = {}
    if param_specs:
        for name, ptype, required, default, desc in param_specs:
            label, placeholder = known_map.get(name, (name.replace("_", " ").title(), ""))
            default_val = default if default is not None else ""
            help_text = desc or None
            if ptype in ("number", "integer"):
                try:
                    v = float(default_val) if default_val != "" else 0.0
                except (TypeError, ValueError):
                    v = 0.0
                params[name] = st.number_input(label, value=v, key=f"ob_{tool_name}_{name}", help=help_text)
                if ptype == "integer":
                    params[name] = int(params[name])
            elif ptype == "boolean":
                params[name] = st.checkbox(label, value=bool(default_val), key=f"ob_{tool_name}_{name}", help=help_text)
            elif ptype == "array" or "array" in str(ptype):
                params[name] = st.text_area(label, value=json.dumps(default_val) if default_val != "" else "[]", key=f"ob_{tool_name}_{name}", height=80, help=help_text or "JSON array, e.g. [\"id1\", \"id2\"]", placeholder=placeholder or None)
                try:
                    params[name] = json.loads(params[name])
                except Exception:
                    pass
            else:
                params[name] = st.text_input(label, value=str(default_val) if default_val != "" else "", key=f"ob_{tool_name}_{name}", help=help_text, placeholder=placeholder or None)
    else:
        st.caption("No parameter schema for this tool. Enter JSON below if you know the parameter names.")
        raw = st.text_area("Parameters (JSON object)", value="{}", key=f"ob_params_{tool_name}", height=100, placeholder='{"query": "your search", "max_results": 10}')
        try:
            params = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            st.error("Invalid JSON")
            return
    st.markdown("")
    if st.button("Run", type="primary", key=f"ob_run_{tool_name}"):
        with st.spinner("Calling OpenBio…"):
            ok, out = openbio_invoke(tool_name, params)
        st.session_state["_openbio_last_result"] = (ok, out, tool_name, params)
        st.rerun()

    # Show last result for this tool (or any) below the form
    last = st.session_state.get("_openbio_last_result")
    if last and last[2] == tool_name:
        ok, out, _ = last[0], last[1], last[2]
        sent_params = last[3] if len(last) > 3 else None
        st.markdown("**Results**")
        if ok:
            if isinstance(out, dict) and "result" in out:
                out = out["result"]
            table_rows = _openbio_result_to_table(out)
            if table_rows:
                st.dataframe(table_rows, use_container_width=True, hide_index=True)
            else:
                st.json(out)
            if isinstance(out, dict):
                pdb_id = out.get("pdb_id") or out.get("pdbId")
                url = out.get("download_url") or out.get("url") or out.get("pdb_url")
                if pdb_id:
                    if st.button("Load this structure in 3D viewer", key=f"ob_load_{tool_name}_pdb"):
                        try:
                            st.session_state.pdb_data = fetch_pdb(str(pdb_id).upper())
                            st.session_state.pdb_source = "pdb_id"
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                elif url:
                    if st.button("Load from URL in 3D viewer", key=f"ob_load_{tool_name}_url"):
                        try:
                            import requests as req
                            r = req.get(url, timeout=60)
                            r.raise_for_status()
                            st.session_state.pdb_data = r.text
                            st.session_state.pdb_source = "upload"
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            elif isinstance(out, list) and out and isinstance(out[0], dict):
                for i, row in enumerate(out[:15]):
                    pdb_id = row.get("pdb_id") or row.get("pdbId") or row.get("id")
                    if pdb_id and st.button(f"Load {pdb_id} in viewer", key=f"ob_load_{tool_name}_{i}_{pdb_id}"):
                        try:
                            st.session_state.pdb_data = fetch_pdb(str(pdb_id).upper())
                            st.session_state.pdb_source = "pdb_id"
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
        else:
            st.error(out)
            if sent_params is not None:
                with st.expander("What we sent (for debugging)"):
                    st.json(sent_params)
                    st.caption("Empty values were omitted before calling the API.")


def _render_openbio_tab(fetch_pdb) -> None:
    """Render OpenBio panel: clear 3-step flow (Category → Tool → Inputs & Run → Results)."""
    st.subheader("OpenBio")
    st.caption(
        "Use the steps below: pick a **category**, then a **tool** (with a short description), "
        "fill the **inputs**, and click **Run this tool**. Results appear below. "
        "API key in `.streamlit/secrets.toml` or **OPENBIO_API_KEY**."
    )
    api_key = openbio_get_api_key()
    if not api_key:
        st.warning("**OPENBIO_API_KEY** is not set. Add it to `.streamlit/secrets.toml` or set the environment variable, then rerun.")
        return
    st.success("API key loaded.")

    # Step 1: Choose category
    st.markdown("#### Step 1 — Choose category")
    cat_options = [f"{c[1]}: {c[2]}" for c in OPENBIO_CATEGORIES]
    cat_choice = st.selectbox(
        "Category",
        range(len(OPENBIO_CATEGORIES)),
        format_func=lambda i: cat_options[i],
        key="openbio_cat",
    )
    cat_id, cat_title, cat_desc, default_tools = OPENBIO_CATEGORIES[cat_choice]
    cat_tools = default_tools

    # Step 2: Choose tool (with clear labels)
    st.markdown("#### Step 2 — Choose tool")
    tool_options = [(t, _openbio_tool_label(t)) for t in cat_tools]
    tool_choice = st.selectbox(
        "Tool",
        range(len(tool_options)),
        format_func=lambda i: tool_options[i][1],
        key="openbio_tool_sel",
    )
    tool_name = tool_options[tool_choice][0]

    # Step 3: Tool form and run
    st.markdown("#### Step 3 — Inputs and run")
    display_name = OPENBIO_TOOL_DESCRIPTIONS.get(tool_name, (tool_name.replace("_", " ").title(), ""))[0]
    st.markdown(f"**Tool:** {display_name}")
    try:
        _openbio_render_tool_form(tool_name, fetch_pdb)
    except Exception as e:
        st.error(str(e))

    # Optional: link to browse by name (advanced)
    with st.expander("Advanced — Run any tool by name (if you know the API tool name)"):
        any_tool = st.text_input("Tool name", placeholder="e.g. fetch_pdb_metadata", key="openbio_any_name")
        if any_tool and any_tool.strip():
            st.caption(f"Running: {any_tool.strip()}")
            try:
                _openbio_render_tool_form(any_tool.strip(), fetch_pdb)
            except Exception as e:
                st.error(str(e))


def main() -> None:
    st.set_page_config(
        page_title="PyMOL3D – Protein & Ligand Viewer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Sidebar help panel
    with st.sidebar.expander("Help & how to use", expanded=False):
        st.markdown(HELP_MARKDOWN)

    # Session state for PDB data and options
    if "pdb_data" not in st.session_state:
        st.session_state.pdb_data = None
    if "pdb_source" not in st.session_state:
        st.session_state.pdb_source = None  # "pdb_id" | "upload"
    if "current_chain" not in st.session_state:
        st.session_state.current_chain = None

    # Handle Update 3D from 2D editor: read smiles_3d from URL and regenerate 3D (before any branch)
    def _get_smiles_3d_param() -> Optional[str]:
        raw = None
        qp = getattr(st, "query_params", None)
        if qp is not None and hasattr(qp, "get"):
            raw = qp.get("smiles_3d")
        if raw is None and hasattr(st, "experimental_get_query_params"):
            try:
                eq = st.experimental_get_query_params()
                raw = eq.get("smiles_3d", [None])[0] if isinstance(eq.get("smiles_3d"), list) else eq.get("smiles_3d")
            except Exception:
                pass
        if raw is None:
            return None
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if not raw:
            return None
        s = str(raw).strip()
        if not s:
            return None
        try:
            s = unquote(s)
        except Exception:
            pass
        return s.strip() or None

    _smiles_3d = _get_smiles_3d_param()
    if _smiles_3d:
        if st.session_state.get("editor_3d_smiles") == _smiles_3d:
            try:
                st.query_params["smiles_3d"] = []
            except Exception:
                pass
            st.rerun()
        else:
            with st.spinner("Generating 3D…"):
                _new_mol_block = smiles_to_3d_mol_block(_smiles_3d)
            if _new_mol_block:
                st.session_state["editor_3d_smiles"] = _smiles_3d
                st.session_state["editor_3d_mol_html"] = _make_3d_molecule_viewer_html(
                    _new_mol_block, width=450, height=560
                )
                st.session_state["2d_display_smiles"] = _smiles_3d
                if "smiles_2d" in st.session_state:
                    st.session_state["smiles_2d"] = _smiles_3d
            try:
                st.query_params["smiles_3d"] = []
            except Exception:
                pass
            st.rerun()

    # Professional workspace CSS: panels, spacing, typography (Maestro/Chimera/PyMOL inspired)
    st.markdown(
        """
    <style>
    /* Full-width workspace */
    .main .block-container { max-width: 100%%; padding: 0.4rem 0.8rem 1rem; }
    /* Two columns: viewer (left) + tools sidebar (right) */
    div[data-testid="column"] {
        padding: 0 8px;
        border-radius: 6px;
    }
    div[data-testid="column"]:first-child {
        padding: 4px 12px 12px 4px;
    }
    div[data-testid="column"]:last-child {
        min-width: 320px;
        background: linear-gradient(180deg, #f8fafc 0%%, #f1f5f9 100%%);
        border-left: 1px solid #e2e8f0;
        padding: 12px 14px 16px;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    div[data-testid="column"]:last-child label, div[data-testid="column"]:last-child p, div[data-testid="column"]:last-child .stCaption {
        font-family: Arial, sans-serif !important;
        font-size: 14px !important;
    }
    /* Section headers: compact, consistent */
    h3 { font-size: 0.95rem; font-weight: 600; color: #334155; margin-top: 0.5rem; margin-bottom: 0.4rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
    /* Expanders: tighter */
    .streamlit-expanderHeader { font-size: 0.9rem; }
    /* Reduce vertical gaps between widgets in sidebars */
    .element-container { margin-bottom: 0.25rem; }
    /* Tabs: compact */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 6px 12px; font-size: 0.9rem; }

    /* Tables: Arial 12/14/16, wide columns, readable */
    .main table, .main div[data-testid="stTable"] table {
        font-family: Arial, sans-serif !important;
        border-collapse: collapse !important;
        width: 100%% !important;
        min-width: 320px;
        table-layout: auto;
        font-size: 12px;
        background: #fff;
    }
    .main table th, .main table td,
    .main div[data-testid="stTable"] th, .main div[data-testid="stTable"] td {
        padding: 12px 16px !important;
        border: 1px solid #e2e8f0 !important;
        text-align: left;
        font-family: Arial, sans-serif !important;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .main table th:first-child, .main table td:first-child,
    .main div[data-testid="stTable"] th:first-child, .main div[data-testid="stTable"] td:first-child {
        min-width: 140px;
        width: 40%%;
    }
    .main table th:last-child, .main table td:last-child,
    .main div[data-testid="stTable"] th:last-child, .main div[data-testid="stTable"] td:last-child {
        min-width: 180px;
        width: 60%%;
    }
    .main table thead th, .main table tr:first-child td,
    .main div[data-testid="stTable"] thead th, .main div[data-testid="stTable"] tr:first-child td {
        font-size: 16px !important;
        font-weight: 600 !important;
        background: #f1f5f9 !important;
        color: #1e293b;
    }
    .main table tbody td:first-child,
    .main div[data-testid="stTable"] tbody td:first-child,
    .main table tr:not(:first-child) td:first-child,
    .main div[data-testid="stTable"] tr:not(:first-child) td:first-child {
        font-size: 14px !important;
        font-weight: 500;
        color: #334155;
    }
    .main table tbody td, .main table tr:not(:first-child) td,
    .main div[data-testid="stTable"] tbody td, .main div[data-testid="stTable"] tr:not(:first-child) td {
        font-size: 12px !important;
    }
    .main table tr:not(:first-child) td:last-child,
    .main div[data-testid="stTable"] tr:not(:first-child) td:last-child {
        font-size: 12px !important;
        color: #475569;
    }
    .main table tr:hover:not(:first-child) td,
    .main div[data-testid="stTable"] tr:hover:not(:first-child) td {
        background: #f8fafc;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Top bar: app title + mode switch (single row)
    _qp = getattr(st, "query_params", None)
    _mode_param = _qp.get("viewer_mode") if _qp and hasattr(_qp, "get") else None
    if isinstance(_mode_param, list):
        _mode_param = _mode_param[0] if _mode_param else None
    _radio_index = 1 if _mode_param == "2D viewer" else 0
    head_left, head_center, head_right = st.columns([6, 1, 2])
    with head_left:
        st.markdown("### 🧬 PyMOL3D — Structure & molecule viewer")
    with head_right:
        viewer_mode = st.radio(
            "Mode",
            options=["3D viewer", "2D viewer"],
            index=_radio_index,
            key="viewer_mode",
            horizontal=True,
            help="3D: protein/ligand. 2D: molecule editor + 3D.",
            label_visibility="collapsed",
        )
    use_3d_viewer = viewer_mode == "3D viewer"
    st.markdown("---")

    # --- Two columns: viewer (left/top) + tools sidebar (right) ---
    col_viewer, col_tools = st.columns([VIEWER_COL_WEIGHT, TOOLS_COL_WEIGHT])

    if use_3d_viewer:
        # --- 3D mode: tools sidebar (right) = Display + Tools + Sequence ---
        with col_tools:
            st.subheader("Display")

            with st.expander("Structure", expanded=True):
                pdb_id_input = st.text_input("PDB ID", placeholder="e.g. 8R4V", key="pdb_id")
                chain_input = st.text_input("Chain (optional)", placeholder="e.g. A", key="chain_input")
                uploaded_file = st.file_uploader("Or upload PDB", type=["pdb", "ent"], key="pdb_upload")
                load_clicked = st.button("Load structure")
                if load_clicked:
                    if uploaded_file is not None:
                        st.session_state.pdb_data = uploaded_file.read().decode("utf-8", errors="replace")
                        st.session_state.pdb_source = "upload"
                    elif pdb_id_input and pdb_id_input.strip():
                        try:
                            st.session_state.pdb_data = fetch_pdb(pdb_id_input.strip())
                            st.session_state.pdb_source = "pdb_id"
                        except Exception as e:
                            st.error(f"Failed to fetch PDB: {e}")
                            st.session_state.pdb_data = None
                    else:
                        st.warning("Enter a PDB ID or upload a PDB file.")
                    st.session_state.current_chain = None
                    st.rerun()

            pdb_data = st.session_state.pdb_data
            chains = get_chains(pdb_data) if pdb_data else []
            ligands = get_ligands(pdb_data) if pdb_data else []
            chain_options = ["All"] + chains if chains else ["All"]
            ligand_options = [f"{ch}:{resn}" for ch, resn in ligands] if ligands else ["None"]
            if not ligand_options:
                ligand_options = ["None"]

            with st.expander("Chain & ligand", expanded=True):
                col_c, col_l = st.columns(2)
                with col_c:
                    chain_sel = st.selectbox("Chain", options=chain_options, index=0, key="chain_sel")
                with col_l:
                    ligand_sel = st.selectbox("Ligand", options=ligand_options, key="ligand_sel")
            ligand_resn = None if not ligand_sel or ligand_sel == "None" else ligand_sel.split(":")[-1]

            with st.expander("Representation", expanded=True):
                preset_options = [p[1] for p in VISUALIZATION_PRESETS]
                preset_sel_label = st.selectbox(
                    "Preset",
                    options=preset_options,
                    index=0,
                    key="viz_preset",
                    help="PyMOL/Chimera/Maestro styles. Custom = use controls below.",
                )
                preset_tuple = next((p for p in VISUALIZATION_PRESETS if p[1] == preset_sel_label), VISUALIZATION_PRESETS[0])
                use_preset = preset_tuple[0] != PRESET_CUSTOM
                col_pc, col_lc = st.columns(2)
                with col_pc:
                    protein_color = st.selectbox("Protein color", options=list(PROTEIN_COLOR_OPTIONS.keys()), key="protein_color")
                    protein_color_val = PROTEIN_COLOR_OPTIONS[protein_color]
                with col_lc:
                    ligand_color = st.selectbox(
                        "Ligand color",
                        options=list(LIGAND_COLOR_OPTIONS.keys()) + [f"Theme: {c.capitalize()}" for c in LIGAND_THEME_COLORS],
                        key="ligand_color",
                        help="Element or theme color.",
                    )
                    if ligand_color in LIGAND_COLOR_OPTIONS:
                        ligand_color_val = LIGAND_COLOR_OPTIONS[ligand_color]
                    elif ligand_color.startswith("Theme: "):
                        ligand_color_val = ligand_color.replace("Theme: ", "").strip().lower()
                    else:
                        ligand_color_val = "element"
                protein_style = st.selectbox("Protein style", options=list(PROTEIN_STYLE_OPTIONS.keys()), key="protein_style")
                protein_style_val = PROTEIN_STYLE_OPTIONS[protein_style]
                protein_opacity = st.slider("Protein opacity", min_value=0.1, max_value=1.0, value=0.85, step=0.05, key="protein_opacity")
                if use_preset:
                    protein_style_val = preset_tuple[2]
                    protein_color_val = preset_tuple[3]
                    protein_opacity = preset_tuple[4]
                    ligand_color_val = preset_tuple[5]
                preset_background = preset_tuple[6] or "0xeeeeee"
                viewer_bg_label = st.selectbox(
                    "3D viewer background",
                    options=[o[0] for o in VIEWER_BACKGROUND_OPTIONS],
                    index=0,
                    key="viewer_bg_3d",
                    help="Background color of the 3D display panel.",
                )
                viewer_bg_hex = next(
                    (o[1] for o in VIEWER_BACKGROUND_OPTIONS if o[0] == viewer_bg_label),
                    "0xeeeeee",
                )

            with st.expander("Analysis", expanded=False):
                show_hbonds = st.checkbox("Show H-bonds (ligand–protein)", value=False, key="show_hbonds")
                hbond_cutoff = st.slider("H-bond cutoff (Å)", min_value=2.5, max_value=4.0, value=3.5, step=0.1, key="hbond_cutoff")
                focus_on_binding_site = st.checkbox("Focus on binding site", value=False, key="focus_binding_site")
                binding_site_radius = 5.0
                hide_distant_protein = False
                binding_site_style = "stick"
                if focus_on_binding_site and ligand_resn:
                    binding_site_radius = st.number_input("Binding site radius (Å)", min_value=3.0, max_value=15.0, value=5.0, step=0.5, key="binding_site_radius")
                    hide_distant_protein = st.checkbox("Hide distant protein", value=False, key="hide_distant_protein")
                    binding_site_style = st.radio("Binding site style", options=["stick", "line"], index=0, key="binding_site_style", format_func=lambda x: "Sticks" if x == "stick" else "Lines", horizontal=True)
                    show_binding_site_labels = st.checkbox("Show residue labels", value=False, key="show_binding_site_labels", help="Label binding site residues (e.g. Lys142). Bold Arial; text color follows 3D viewer background.")
                    if show_binding_site_labels:
                        label_font_size = st.selectbox("Label font size", options=[10, 12, 14, 16, 18, 20], index=2, key="label_font_size_3d", format_func=lambda x: f"{x} px")
                    else:
                        label_font_size = 14
                else:
                    show_binding_site_labels = False
                    label_font_size = 14
                show_pharmacophore = st.checkbox("Show 3D pharmacophore", value=False, key="show_pharmacophore")
                pharmacophore_sphere_radius = 0.5
                pharmacophore_feature_set = "BaseFeatures"
                if show_pharmacophore:
                    pharmacophore_feature_set = st.radio("Feature set", options=["BaseFeatures", "Gobbi"], index=0, key="pharmacophore_feature_set", horizontal=True)
                    pharmacophore_sphere_radius = st.number_input("Sphere radius (Å)", min_value=0.2, max_value=2.0, value=0.5, step=0.1, key="pharmacophore_sphere_radius")

            # Tools: residues, Protein sequence (2D molecule, descriptors, fingerprints are in 2D viewer only)
            st.subheader("Tools")
            if pdb_data:
                with st.expander("Residues to focus", expanded=False):
                    res_list = get_residue_list_for_selector(pdb_data, chain_sel if chain_sel != "All" else None)
                    res_options = [r[0] for r in res_list]
                    res_label_to_key = {r[0]: r[1] for r in res_list}
                    focus_labels = st.multiselect("Highlight + sticks", options=res_options, default=[], key="focus_res")
                    focus_res_ids = [res_label_to_key[l] for l in focus_labels if l in res_label_to_key]
            else:
                focus_res_ids = []

            # Protein sequence
            if pdb_data:
                with st.expander("Protein sequence", expanded=True):
                    seq = sequence_string(pdb_data, chain_sel if chain_sel != "All" else None)
                    if seq:
                        st.text(seq)
                        st.caption("Use «Residues to focus» above to highlight one or more residues in the 3D view.")
                    else:
                        st.caption("No sequence for selected chain.")

        # Resolve chain for H-bond detection (after widgets so chain_sel/ligand_sel are set)
        hbond_chain = None
        if pdb_data and ligand_resn:
            hbond_chain = chain_sel if chain_sel != "All" else (ligand_sel.split(":")[0].strip() if ligand_sel and ligand_sel != "None" else None)
        hbond_pairs = []
        if pdb_data and show_hbonds and ligand_resn and hbond_chain:
            try:
                hbond_pairs = find_hbonds(pdb_data, hbond_chain, ligand_resn, cutoff=hbond_cutoff)
            except Exception:
                hbond_pairs = []

        # --- Viewer (left): main Viewer tab ---
        with col_viewer:
            tab_viewer_main, = st.tabs(["Viewer"])
            with tab_viewer_main:
                if pdb_data:
                    pdb_block = get_pdb_for_chain(pdb_data, chain_sel if chain_sel != "All" else None)
                    viewer_width = 1100
                    viewer_height = 650
                    binding_site_res_ids = None
                    if focus_on_binding_site and ligand_resn and hbond_chain:
                        binding_site_res_ids = get_binding_site_residues(
                            pdb_block, ligand_resn, hbond_chain, distance_cutoff=binding_site_radius
                        )
                    binding_site_labels = None
                    if show_binding_site_labels and binding_site_res_ids:
                        binding_site_labels = get_residue_labels(pdb_block, binding_site_res_ids)
                    pharmacophore_points = None
                    if show_pharmacophore and ligand_resn and hbond_chain and pdb_data:
                        ligand_block = get_ligand_pdb_block(pdb_data, ligand_resn, hbond_chain)
                        pharmacophore_points = get_pharmacophore_points(
                            ligand_block, feature_set=pharmacophore_feature_set
                        )
                    html = _make_viewer_html(
                        pdb_block=pdb_block,
                        protein_style=protein_style_val,
                        protein_color=protein_color_val,
                        protein_opacity=protein_opacity,
                        ligand_resn=ligand_resn,
                        ligand_color=ligand_color_val,
                        focus_res_ids=focus_res_ids,
                        width=viewer_width,
                        height=viewer_height,
                        hbond_pairs=hbond_pairs if show_hbonds else None,
                        binding_site_res_ids=binding_site_res_ids,
                        hide_distant_protein=hide_distant_protein if focus_on_binding_site else False,
                        binding_site_style=binding_site_style,
                        background_color=viewer_bg_hex,
                        pharmacophore_points=pharmacophore_points,
                        pharmacophore_sphere_radius=pharmacophore_sphere_radius,
                        binding_site_labels=binding_site_labels,
                        label_font_size=label_font_size,
                    )
                    tab_viewer, tab_hbonds, tab_plip = st.tabs(["3D viewer", "Hydrogen bonds", "2D interaction plot (PLIP)"])
                    with tab_viewer:
                        st.components.v1.html(html, height=viewer_height + 40, scrolling=False)
                        st.caption("Rotate/zoom with mouse. Full screen: button in viewer; Escape to exit.")
                    with tab_hbonds:
                        st.subheader("Hydrogen bonds")
                        if ligand_resn and hbond_chain:
                            if show_hbonds:
                                st.success(f"**{len(hbond_pairs)}** potential H-bonds shown (distance ≤ {hbond_cutoff} Å). Yellow dashed lines in the **3D viewer** tab.")
                            else:
                                st.info("Enable **Show H-bonds (ligand–protein)** in the right panel, then switch to the **3D viewer** tab to see dashed bonds between the ligand and protein residues.")
                        else:
                            st.warning("Select a chain and a ligand (other than «None») to visualize hydrogen bonds.")
                        st.caption("H-bonds are detected by heavy-atom distance (N/O). Use the **3D viewer** tab to see the structure with bonds when the option is on.")

                    with tab_plip:
                        st.subheader("2D interaction plot (PLIP)")
                        if ligand_resn and pdb_data:
                            st.caption("Uses [PLIP](https://github.com/pharmai/plip) to detect H-bonds, hydrophobic, pi-stacking, salt bridges, and other interactions. Optional **plotly** for an interactive plot; **matplotlib** fallback.")
                            plip_display_w, plip_display_h = 600, 500
                            gen_plip = st.button("Generate PLIP 2D plot", key="gen_plip_2d")
                            if gen_plip:
                                with st.spinner("Running PLIP and building plot…"):
                                    pdb_for_plip = get_pdb_for_chain(pdb_data, chain_sel if chain_sel != "All" else None)
                                    plip_chain = hbond_chain if ligand_resn and (chain_sel == "All" or chain_sel == hbond_chain) else (chain_sel if chain_sel != "All" else None)
                                    plip_fig, png_bytes, plip_err = generate_plip_2d_plot(
                                        pdb_for_plip,
                                        ligand_resn,
                                        chain_id=plip_chain,
                                        width=plip_display_w,
                                        height=plip_display_h,
                                    )
                                    st.session_state["plip_2d_fig"] = plip_fig
                                    st.session_state["plip_2d_png"] = png_bytes
                                    st.session_state["plip_2d_error"] = plip_err
                                    st.session_state["plip_pdb"] = pdb_for_plip
                                    st.session_state["plip_ligand"] = ligand_resn
                                    st.session_state["plip_chain"] = plip_chain
                                st.rerun()
                            plip_fig = st.session_state.get("plip_2d_fig")
                            plip_png = st.session_state.get("plip_2d_png")
                            plip_err = st.session_state.get("plip_2d_error")
                            if plip_err:
                                st.error(plip_err)
                            elif plip_fig is not None or plip_png:
                                if plip_fig is not None:
                                    st.plotly_chart(plip_fig, width="stretch")
                                elif plip_png:
                                    st.image(plip_png, width="stretch")
                                if st.button("Download PNG", key="dl_plip_btn"):
                                    _open_download_plip_dialog()
                            else:
                                st.info("Click **Generate PLIP 2D plot** to run PLIP and show residue–interaction plot.")
                        else:
                            st.warning("Load a structure and select a ligand (other than «None») to generate the PLIP 2D plot.")
                        st.caption("Requires **plip** (and **matplotlib** or **plotly**). PLIP needs **OpenBabel** first — e.g. `conda install -c conda-forge openbabel` then `pip install openbabel plip matplotlib`.")
                else:
                    st.info("Load a structure (right panel: Structure → PDB ID or upload) then **Load structure**.")

    else:
        # --- 2D viewer mode: Left = Sequence & tools, Center = MolView (main window), Right = options ---
        # Use smiles_2d in 2D mode to avoid Streamlit "cannot modify after widget" error.
        # Only touch session_state["smiles_2d"] here at the start of the block, before any widget with key smiles_2d is created.
        if "smiles_2d" not in st.session_state:
            st.session_state["smiles_2d"] = (
                st.session_state.get("smiles_3d") or st.session_state.get("smiles") or ""
            )
            if "2d_display_smiles" not in st.session_state:
                st.session_state["2d_display_smiles"] = st.session_state["smiles_2d"]
        if st.session_state.pop("smiles_2d_sync_from_lookup", None) and st.session_state.get("2d_display_smiles") is not None:
            st.session_state["smiles_2d"] = st.session_state["2d_display_smiles"]
        sidebar_smiles_for_editor = (
            (st.session_state.get("2d_display_smiles") or st.session_state.get("smiles_2d") or "").strip()
        )
        with col_tools:
            st.subheader("Options")
            st.caption("2D editor (left) and 3D view (right) share the same molecule. Use **Update 3D** in the viewer to refresh 3D from the current sketch.")
            opt_col_bg, opt_col_rep = st.columns(2)
            with opt_col_bg:
                viewer_bg_2d_label = st.selectbox(
                    "3D panel background",
                    options=[o[0] for o in VIEWER_BACKGROUND_OPTIONS],
                    index=0,
                    key="viewer_bg_2d",
                    help="Background color of the 3D display panel (right side).",
                )
                viewer_bg_2d_css = next(
                    (o[2] for o in VIEWER_BACKGROUND_OPTIONS if o[0] == viewer_bg_2d_label),
                    "#eeeeee",
                )
            with opt_col_rep:
                representation_2d_label = st.selectbox(
                    "3D representation",
                    options=[o[0] for o in REPRESENTATION_2D_OPTIONS],
                    index=0,
                    key="representation_2d",
                    help="Sticks, lines, or licorice (thick sticks) in the 3D display.",
                )
                representation_2d_val = next(
                    (o[1] for o in REPRESENTATION_2D_OPTIONS if o[0] == representation_2d_label),
                    "stick",
                )
            thickness_2d = st.slider(
                "Molecule thickness",
                min_value=0.08,
                max_value=0.5,
                value=0.2,
                step=0.02,
                key="thickness_2d",
                help="Thickness of bonds and atoms in the 3D display (stick radius).",
            )
            st.subheader("Tools")
            with st.expander("2D molecule", expanded=True):
                mol_name_input_2d = st.text_input(
                    "Molecule name (optional)",
                    placeholder="e.g. aspirin, caffeine",
                    key="mol_name_2d",
                    help="Look up from PubChem or ChEMBL to fill SMILES and show basic info.",
                )
                lookup_clicked_2d = st.button("Look up", key="mol_lookup_2d")
                mol_info = st.session_state.get("mol_info")
                if lookup_clicked_2d and mol_name_input_2d and mol_name_input_2d.strip():
                    with st.spinner("Looking up molecule…"):
                        info = lookup_molecule_by_name(mol_name_input_2d.strip())
                    if info:
                        st.session_state["mol_info"] = info
                        sm = info.get("smiles", "") or st.session_state.get("smiles_2d", "")
                        st.session_state["2d_display_smiles"] = sm
                        st.session_state["smiles_2d"] = sm
                        st.session_state["smiles_2d_sync_from_lookup"] = True
                        st.rerun()
                    else:
                        st.warning(f"No molecule found for «{mol_name_input_2d.strip()}» in PubChem or ChEMBL.")
                if mol_info:
                    st.markdown("**Basic information**")
                    rows = []
                    if mol_info.get("title"):
                        rows.append(("Name", mol_info["title"]))
                    if mol_info.get("smiles"):
                        rows.append(("SMILES", mol_info["smiles"]))
                    if mol_info.get("formula"):
                        rows.append(("Molecular formula", mol_info["formula"]))
                    if mol_info.get("molecular_weight"):
                        rows.append(("Molecular weight", str(mol_info["molecular_weight"])))
                    if mol_info.get("iupac_name"):
                        rows.append(("IUPAC name", mol_info["iupac_name"]))
                    if mol_info.get("source"):
                        rows.append(("Source", mol_info["source"]))
                    if mol_info.get("cid"):
                        rows.append(("PubChem CID", str(mol_info["cid"])))
                    if mol_info.get("chembl_id"):
                        rows.append(("ChEMBL ID", mol_info["chembl_id"]))
                    if rows:
                        st.table([["Property", "Value"]] + rows)
                    st.session_state["mol_info"] = mol_info

                smiles_input_2d = st.text_input(
                    "SMILES",
                    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O or fill via Look up",
                    key="smiles_2d",
                )
                col_draw, col_load = st.columns(2)
                with col_draw:
                    draw_2d_2d = st.button("Draw", key="draw_2d_2d", help="Preview 2D structure")
                with col_load:
                    load_into_editor_btn = st.button("Load into editor", key="draw_2d_center", help="Open this molecule in the main view for drawing or editing")
                if load_into_editor_btn and smiles_input_2d and smiles_input_2d.strip():
                    st.session_state["2d_display_smiles"] = smiles_input_2d.strip()
                    st.rerun()
                if draw_2d_2d and smiles_input_2d and HAS_RDKIT:
                    png_bytes_2d = render_2d_mol(smiles_input_2d, size=(500, 500))
                    if png_bytes_2d:
                        st.image(png_bytes_2d, width="stretch")
                    else:
                        st.warning("Could not parse SMILES.")
                elif draw_2d_2d and not HAS_RDKIT:
                    st.warning("RDKit not installed. Install with: pip install rdkit")
                if HAS_RDKIT:
                    if st.button("Download PNG", key="open_dl_2d_btn_2d"):
                        _open_download_2d_dialog()
                    if not (getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)):
                        with st.expander("Download PNG (set resolution)"):
                            st.caption("Popup requires Streamlit 1.33+. Set resolution here:")
                            _cw, _ch = st.columns(2)
                            with _cw:
                                _dl_w_2d = st.number_input("Width (px)", min_value=200, max_value=2400, value=800, key="dl_2d_fb_w_2d")
                            with _ch:
                                _dl_h_2d = st.number_input("Height (px)", min_value=200, max_value=2400, value=800, key="dl_2d_fb_h_2d")
                            _sm_2d = (st.session_state.get("smiles_2d") or "").strip()
                            _png_2d = render_2d_mol(_sm_2d, (_dl_w_2d, _dl_h_2d)) if _sm_2d else None
                            if _png_2d:
                                st.download_button("Save PNG", data=_png_2d, file_name="molecule_2d.png", mime="image/png", key="dl_2d_fb_btn_2d")
                            else:
                                st.caption("Enter SMILES above first.")
                st.caption("Paste SMILES, click **Draw** to preview or **Load into editor** to open in the main view.")

            with st.expander("Molecular descriptors", expanded=False):
                if HAS_RDKIT:
                    if "mol_descriptor_table" not in st.session_state:
                        st.session_state["mol_descriptor_table"] = {}
                    tab_single, tab_batch = st.tabs(["Single molecule", "Batch (CSV)"])

                    # --- Single-molecule descriptors (including ESOL LogS) ---
                    with tab_single:
                        selected_descriptors = st.multiselect(
                            "Properties to calculate",
                            options=RDKIT_DESCRIPTOR_NAMES,
                            default=[],
                            key="descriptor_multiselect_2d",
                            help="Select one or more RDKit descriptors; computed for the current SMILES (sidebar or from editor).",
                        )
                        col_calc, col_clear = st.columns(2)
                        with col_calc:
                            calc_clicked = st.button("Calculate property", key="descriptor_calc_btn_2d")
                        with col_clear:
                            clear_clicked = st.button("Clear table", key="descriptor_clear_btn_2d")
                        if calc_clicked:
                            smiles = (st.session_state.get("smiles_2d") or "").strip()
                            if not smiles:
                                st.warning("Enter SMILES above or apply from editor first.")
                            else:
                                mol = Chem.MolFromSmiles(smiles)
                                if mol is None:
                                    st.warning("Invalid SMILES.")
                                elif not selected_descriptors:
                                    st.warning("Select at least one property.")
                                else:
                                    try:
                                        all_vals = Descriptors.CalcMolDescriptors(mol)
                                        if "QED" in selected_descriptors:
                                            all_vals["QED"] = QED.qed(mol)
                                        # ESOL solubility (logS) and component properties
                                        esol_log_s, esol_logp, esol_mw, esol_rb, esol_ap = _compute_esol_from_mol(mol)
                                        if esol_log_s is not None:
                                            all_vals["ESOL_LogS"] = esol_log_s
                                            all_vals["ESOL_MolLogP"] = esol_logp
                                            all_vals["ESOL_MolWt"] = esol_mw
                                            all_vals["ESOL_NumRotBonds"] = esol_rb
                                            all_vals["ESOL_AromaticProportion"] = esol_ap
                                        for k in selected_descriptors:
                                            if k in all_vals:
                                                st.session_state["mol_descriptor_table"][k] = all_vals[k]
                                        st.rerun()
                                    except Exception as e:
                                        st.warning(f"Descriptor calculation failed: {e}")
                        if clear_clicked:
                            st.session_state["mol_descriptor_table"] = {}
                            st.rerun()
                        if st.session_state["mol_descriptor_table"]:
                            rows = [[str(k), _table_value_str(v)] for k, v in st.session_state["mol_descriptor_table"].items()]
                            st.table([["Property", "Value"]] + rows)
                        else:
                            st.caption("Select properties above, then click **Calculate property**.")

                        # Download CSV: basic info + descriptors
                        _mol_info = st.session_state.get("mol_info")
                        _desc_table = st.session_state.get("mol_descriptor_table") or {}
                        if _mol_info or _desc_table:
                            csv_data = _molecule_info_and_properties_csv(_mol_info, _desc_table)
                            st.download_button(
                                "Download CSV (basic info + properties)",
                                data=csv_data,
                                file_name="molecule_info_and_properties.csv",
                                mime="text/csv",
                                key="dl_mol_csv_btn_2d",
                            )

                    # --- Batch descriptors from CSV: QED + ESOL LogS and components ---
                    with tab_batch:
                        st.caption("Upload a CSV with a SMILES column to compute QED, ESOL LogS (log10 molar solubility), and its component properties in batch.")
                        batch_file = st.file_uploader(
                            "CSV file with SMILES column",
                            type=["csv"],
                            key="batch_desc_upload",
                        )
                        smiles_col_name = st.text_input(
                            "SMILES column name",
                            value="SMILES",
                            key="batch_desc_smiles_col",
                            help="Name of the column in the uploaded CSV that contains SMILES strings.",
                        )
                        run_batch = st.button("Compute batch descriptors", key="batch_desc_run")
                        if run_batch:
                            if not batch_file:
                                st.warning("Upload a CSV file first.")
                            elif not smiles_col_name.strip():
                                st.warning("Enter the SMILES column name.")
                            else:
                                try:
                                    df_batch = pd.read_csv(batch_file)
                                except Exception as e:
                                    st.warning(f"Could not read CSV: {e}")
                                    df_batch = None
                                if df_batch is not None:
                                    col = smiles_col_name.strip()
                                    if col not in df_batch.columns:
                                        st.warning(f"Column «{col}» not found in CSV.")
                                    else:
                                        smiles_series = df_batch[col]
                                        esol_logs: list[Optional[float]] = []
                                        esol_logps: list[Optional[float]] = []
                                        esol_mws: list[Optional[float]] = []
                                        esol_rbs: list[Optional[float]] = []
                                        esol_aps: list[Optional[float]] = []
                                        qeds: list[Optional[float]] = []
                                        statuses: list[str] = []
                                        with st.spinner("Computing descriptors…"):
                                            for sm in smiles_series:
                                                smi = str(sm) if not pd.isna(sm) else ""
                                                if not smi.strip():
                                                    esol_logs.append(None)
                                                    esol_logps.append(None)
                                                    esol_mws.append(None)
                                                    esol_rbs.append(None)
                                                    esol_aps.append(None)
                                                    qeds.append(None)
                                                    statuses.append("Empty SMILES")
                                                    continue
                                                mol = Chem.MolFromSmiles(smi.strip())
                                                if mol is None:
                                                    esol_logs.append(None)
                                                    esol_logps.append(None)
                                                    esol_mws.append(None)
                                                    esol_rbs.append(None)
                                                    esol_aps.append(None)
                                                    qeds.append(None)
                                                    statuses.append("Invalid SMILES")
                                                    continue
                                                es_log_s, es_logp, es_mw, es_rb, es_ap = _compute_esol_from_mol(mol)
                                                try:
                                                    q_val = float(QED.qed(mol))
                                                except Exception:
                                                    q_val = None
                                                esol_logs.append(es_log_s)
                                                esol_logps.append(es_logp)
                                                esol_mws.append(es_mw)
                                                esol_rbs.append(es_rb)
                                                esol_aps.append(es_ap)
                                                qeds.append(q_val)
                                                statuses.append("OK" if es_log_s is not None else "ESOL failed")
                                        out_df = df_batch.copy()
                                        out_df["QED"] = qeds
                                        out_df["ESOL_LogS"] = esol_logs
                                        out_df["ESOL_MolLogP"] = esol_logps
                                        out_df["ESOL_MolWt"] = esol_mws
                                        out_df["ESOL_NumRotBonds"] = esol_rbs
                                        out_df["ESOL_AromaticProportion"] = esol_aps
                                        out_df["DescriptorStatus"] = statuses
                                        st.success(f"Computed descriptors for {len(out_df)} rows.")
                                        st.dataframe(out_df, use_container_width=True)
                                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                                        st.download_button(
                                            "Download results CSV",
                                            data=csv_bytes,
                                            file_name="batch_descriptors.csv",
                                            mime="text/csv",
                                            key="batch_desc_download",
                                        )
                else:
                    st.caption("Install **rdkit** to calculate molecular descriptors and download CSV.")

            with st.expander("Fingerprints", expanded=False):
                if not HAS_RDKIT:
                    st.caption("Install **rdkit** to compute fingerprints.")
                else:
                    st.caption("Upload a .smi or .txt file: one SMILES per line, no header.")
                    fp_file_2d = st.file_uploader(
                        "SMILES file",
                        type=["smi", "txt"],
                        key="fp_upload_2d",
                    )
                    fp_type_label_2d = st.selectbox(
                        "Fingerprint type",
                        options=[o[0] for o in FINGERPRINT_OPTIONS],
                        key="fp_type_2d",
                    )
                    fp_key_2d = next((o[1] for o in FINGERPRINT_OPTIONS if o[0] == fp_type_label_2d), "rdkit")
                    fp_compute_2d = st.button("Compute fingerprints", key="fp_compute_2d")
                    if fp_compute_2d and fp_file_2d:
                        raw = fp_file_2d.read().decode("utf-8", errors="replace")
                        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                        if not lines:
                            st.warning("File is empty or has no valid lines.")
                        else:
                            with st.spinner("Computing…"):
                                fp_rows_2d = _compute_fingerprints_for_smiles_list(lines, fp_key_2d)
                            st.session_state["fp_results_2d"] = (fp_type_label_2d, fp_rows_2d)
                            st.rerun()
                    if st.session_state.get("fp_results_2d"):
                        _label2, _rows2 = st.session_state["fp_results_2d"]
                        st.success(f"**{len(_rows2)}** molecules, type: {_label2}")
                        if _rows2:
                            st.dataframe(
                                [{"SMILES": r[0], "Status": r[1], "Fingerprint": r[2][:80] + "…" if len(r[2]) > 80 else r[2]} for r in _rows2],
                                use_container_width=True,
                                height=min(300, 50 + 35 * min(len(_rows2), 8)),
                            )
                            csv_out_2d = _fingerprints_to_csv(_rows2, _label2)
                            st.download_button(
                                "Download as CSV",
                                data=csv_out_2d,
                                file_name="fingerprints.csv",
                                mime="text/csv",
                                key="dl_fp_csv_2d",
                            )

        # Viewer (left): editor + 3D window
        with col_viewer:
            total_width = 900
            total_height = 580
            mol_block = None
            # Auto-load 3D from 2D editor when there is SMILES in the editor but no 3D shown yet
            if sidebar_smiles_for_editor and not st.session_state.get("editor_3d_mol_html"):
                with st.spinner("Loading 3D from 2D editor…"):
                    auto_mol_block = smiles_to_3d_mol_block(sidebar_smiles_for_editor)
                if auto_mol_block:
                    st.session_state["editor_3d_smiles"] = sidebar_smiles_for_editor
                    st.session_state["editor_3d_mol_html"] = _make_3d_molecule_viewer_html(
                        auto_mol_block, width=450, height=560
                    )
                    st.rerun()
            if st.session_state.get("editor_3d_mol_html") and st.session_state.get("editor_3d_smiles"):
                mol_block = smiles_to_3d_mol_block(st.session_state["editor_3d_smiles"])
            combined_html = _make_combined_editor_3d_html(
                initial_smiles=sidebar_smiles_for_editor,
                mol_block=mol_block,
                total_width=total_width,
                total_height=total_height,
                background_color_2d=viewer_bg_2d_css,
                representation_2d=representation_2d_val,
                thickness_2d=thickness_2d,
            )
            st.caption("2D editor (left) | 3D view (right). Buttons: Update 3D, Full screen, Reset view.")
            st.components.v1.html(combined_html, height=total_height + 20, scrolling=False)


if __name__ == "__main__":
    main()
