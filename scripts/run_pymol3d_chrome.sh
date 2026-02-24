#!/usr/bin/env bash
# Run the Pymol3D Streamlit app and open in Chrome (or Chromium).
# Usage: ./run_pymol3d_chrome.sh
cd "$(dirname "$0")"
if command -v google-chrome &>/dev/null; then
  export BROWSER='google-chrome'
elif command -v google-chrome-stable &>/dev/null; then
  export BROWSER='google-chrome-stable'
else
  export BROWSER='chromium'
fi
streamlit run pymol3d_app.py --server.headless true
