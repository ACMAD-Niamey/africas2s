#!/usr/bin/env bash
# install_pycpt.sh — Install PyCPT v2.10.4 for AfricaS2S comparison.
#
# Installs miniforge (via brew) if conda is not available, then creates
# the pycpt conda env from the official lock file. CPT binary is x86_64
# only, so we force osx-64 (runs via Rosetta 2 on Apple Silicon).
#
# Usage:
#   bash scripts/install_pycpt.sh
#
# After install:
#   conda activate pycpt
#   python scripts/compare_pycpt.py
set -euo pipefail

PYCPT_VERSION="2.10.4"
ENV_NAME="pycpt"
RELEASE_BASE="https://github.com/iri-pycpt/notebooks/releases/download/v${PYCPT_VERSION}"
LOCK_FILE="conda-osx-64.lock"

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*"; exit 1; }

# --- 1. Ensure conda is available ---
if ! command -v conda &>/dev/null; then
    # Try to source it from common locations
    for p in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi

if ! command -v conda &>/dev/null; then
    info "conda not found. Installing miniforge via brew..."
    if ! command -v brew &>/dev/null; then
        error "Neither conda nor brew found. Install one first."
    fi
    brew install miniforge
    source "$(brew --prefix miniforge)/etc/profile.d/conda.sh"
fi

if ! command -v conda &>/dev/null; then
    error "conda still not available after install. Check your shell config."
fi
info "Found conda: $(conda --version)"

# --- 2. Remove old pycpt env if it exists ---
if conda env list | grep -q "^${ENV_NAME} "; then
    info "Removing existing ${ENV_NAME} environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
fi

# --- 3. Download lock file ---
TMPDIR_DL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_DL"' EXIT

info "Downloading lock file for PyCPT v${PYCPT_VERSION}..."
curl -fSL "${RELEASE_BASE}/${LOCK_FILE}" -o "${TMPDIR_DL}/${LOCK_FILE}"

# --- 4. Create environment ---
info "Creating conda environment '${ENV_NAME}' (this may take a few minutes)..."
CONDA_SUBDIR=osx-64 conda create -n "${ENV_NAME}" --file "${TMPDIR_DL}/${LOCK_FILE}" -y

# --- 5. Pin subdir ---
set +u
conda activate "${ENV_NAME}"
set -u
conda config --env --set subdir osx-64

# --- 6. Verify ---
info "Verifying PyCPT import..."
if python -c "import pycpt; print('pycpt version:', pycpt.__version__)"; then
    info "PyCPT v${PYCPT_VERSION} installed successfully."
    info ""
    info "To use:"
    info "  conda activate pycpt"
    info "  python scripts/compare_pycpt.py"
else
    error "PyCPT import failed."
fi
