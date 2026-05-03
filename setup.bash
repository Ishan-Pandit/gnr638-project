#!/bin/bash
# =============================================================
# GNR638 Project – setup.bash
# Runs WITH internet. Sets up conda env and clones repo.
# =============================================================

set -e   # Exit immediately on error

echo "============================================"
echo " GNR638 Project Setup"
echo "============================================"

# ── 1. Clone the project repository ──────────────────────────
REPO_URL="https://github.com/Ishan-Pandit/gnr638-project.git"
# Replace YOUR_GITHUB_USERNAME with your actual GitHub username before submission.
# Make sure the repository is PUBLIC before 03rd May 2026 11:00 AM.

if [ -d "gnr638-project" ]; then
    echo "[INFO] Repo already cloned, pulling latest..."
    cd gnr638-project && git pull && cd ..
else
    echo "[INFO] Cloning repository..."
    git clone "$REPO_URL" gnr638-project
fi

# Copy inference script to working directory (grader runs from this directory)
cp gnr638-project/inference.py ./inference.py

echo "[INFO] Repository cloned successfully."

# ── 2. Create Conda environment ───────────────────────────────
ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

echo "[INFO] Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

# Remove if already exists (clean slate)
conda remove --name "$ENV_NAME" --all -y 2>/dev/null || true

conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "[INFO] Conda environment created."

# ── 3. Activate and install dependencies ─────────────────────
echo "[INFO] Installing dependencies..."

# Use conda run to install into the env without needing to source conda
conda run -n "$ENV_NAME" pip install --upgrade pip

conda run -n "$ENV_NAME" pip install \
    numpy==1.26.4 \
    pandas==2.2.2 \
    opencv-python-headless==4.9.0.80 \
    scikit-image==0.23.2 \
    scikit-learn==1.4.2 \
    networkx==3.3 \
    matplotlib==3.9.0 \
    scipy==1.13.1 \
    Pillow==10.3.0 \
    tqdm==4.66.4

echo "[INFO] All dependencies installed."

# ── 4. Verify installation ────────────────────────────────────
echo "[INFO] Verifying key packages..."
conda run -n "$ENV_NAME" python -c "
import cv2, numpy, pandas, networkx, sklearn, skimage, scipy
print(f'  opencv  : {cv2.__version__}')
print(f'  numpy   : {numpy.__version__}')
print(f'  pandas  : {pandas.__version__}')
print(f'  networkx: {networkx.__version__}')
print(f'  sklearn : {sklearn.__version__}')
print(f'  skimage : {skimage.__version__}')
print(f'  scipy   : {scipy.__version__}')
print('  [OK] All packages verified.')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo " Run:"
echo "   conda activate gnr_project_env"
echo "   python inference.py --test_dir <path_to_test_dir>"
echo "============================================"
