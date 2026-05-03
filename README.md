# GNR638 – Geospatial Image Stitching & MCQ Answering

## Submission Checklist

- [ ] Replace `YOUR_GITHUB_USERNAME` in `setup.bash` with your actual GitHub username
- [ ] Push all files to GitHub and make the repo **PUBLIC** before 03 May 2026, 11:00 AM
- [ ] Rename the zip as per naming convention (see below)
- [ ] Zip contains **only** `setup.bash`

## Zip Naming Convention

```
project_<num>_<roll1>_<roll2>.zip
```

Example: `project_2_22m2162_22m2152.zip`

## Repository Files

```
gnr638-project/
├── inference.py      ← main inference script
├── README.md
└── (any other helper files)
```

`setup.bash` clones this repo and copies `inference.py` to the grading directory.

## Grading Commands (as specified)

```bash
cd ./your_directory
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

## Test Directory Structure Expected

```
test_dir/
    sample_1/
        patch_1.png
        patch_2.png
        ...
    sample_2/
        ...
    test.csv              ← questions
    submission.csv        ← dummy submission template
```

## Environment

- Name : `gnr_project_env`
- Python: `3.11`
- Target: Linux, L40s GPU (48 GB VRAM), CUDA 12.6, 16 GB RAM
- Inference: **CPU-only, fully offline**

## Pipeline Summary

| Module | Description |
|--------|-------------|
| A – Data Loader      | Auto-detects test structure, reads patches |
| B – Preprocessing    | CLAHE + Gaussian denoise + resize cap |
| C – Patch Matching   | ORB → AKAZE → SIFT fallback, Lowe ratio |
| D – Graph            | Weighted NetworkX + Maximum Spanning Tree |
| E – Homography       | RANSAC → Affine fallback, degenerate check |
| F – Mosaic           | Global transform composition, feather blend |
| G – Quality Check    | Auto-repair with alt roots, grid fallback |
| H – Scene Features   | GI, blue-dom, GLCM, LBP, quadrant stats |
| I – Question Parser  | Regex keyword maps, 8 question types |
| J – MCQ Engine       | Feature→option scoring per question type |
| K – Ensemble         | Weighted vote across 4 independent engines |
