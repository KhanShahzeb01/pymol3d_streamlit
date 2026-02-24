#!/usr/bin/env bash
# Install PLIP and OpenBabel for the 2D interaction plot tab.
# Run from project root: ./pymol3d/install_plip.sh
# Uses openbabel-wheel (pre-built); then installs plip without building openbabel from source.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/venv}"
PIP="$VENV/bin/pip"

if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "Virtual environment not found at $VENV. Create it first: python3 -m venv venv"
  exit 1
fi

echo "1. Installing openbabel-wheel (pre-built OpenBabel, no SWIG needed)..."
"$PIP" install openbabel-wheel

echo "2. Building plip wheel (openbabel will fail to build; plip wheel is still created)..."
"$PIP" install plip --no-build-isolation 2>/dev/null || true

echo "3. Installing plip from the built wheel and lxml..."
PLIP_WHEEL=$(find "$HOME/.cache/pip/wheels" -path "*/plip-*-py3-none-any.whl" -type f 2>/dev/null | head -1)
if [[ -z "$PLIP_WHEEL" ]]; then
  echo "Plip wheel not in cache. Run manually: pip install plip --no-build-isolation (let it fail), then re-run this script."
  exit 1
fi
"$PIP" install "$PLIP_WHEEL" --no-deps
"$PIP" install lxml

echo "4. Checking imports..."
"$VENV/bin/python3" -c "from plip.structure.preparation import PDBComplex; import openbabel; print('PLIP and OpenBabel OK')"
echo "Done."
