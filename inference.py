#!/usr/bin/env python3
"""
GNR638 Project 1 – Geospatial Patch Stitching & MCQ Answering
inference.py

Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>

Expected test_dir layout:
    <test_dir>/
        patches/
            patch_0.png     <- always top-left anchor, never rotated
            patch_1.png     <- may be shuffled and/or rotated 0/90/180/270 deg
            ...
        test.csv            <- columns: question_id, question, option_1..4
        sample_submission.csv

Output (written to CWD, NOT to test_dir):
    ./submission.csv
    columns : id, question_num, option
    option  : 1 / 2 / 3 / 4 (answer) or 5 (unanswered, 0 penalty)
    INVALID values cause -1 penalty, so we never output anything else.

Constraints:
    - Fully OFFLINE (no internet, no external API calls)
    - No GPU needed (pure CPU OpenCV + NumPy)
    - Conda env: gnr_project_env, Python 3.11
"""

import argparse
import os
import re
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
CFG = {
    # Edge-continuity strip width (pixels) for greedy patch placement
    "EDGE_STRIP_W"   : 20,
    # Maximum grid search dimensions
    "MAX_GRID_COLS"  : 14,
    "MAX_GRID_ROWS"  : 14,
    # Segmentation clusters
    "SEG_CLUSTERS"   : 6,
    # Answer confidence gap below which we output 5 (unanswered, 0 penalty)
    # rather than risking -0.25 for a wrong answer
    "CONF_THRESHOLD" : 0.04,
}

VALID_OPTIONS = {1, 2, 3, 4, 5}


# =============================================================================
# MODULE A – Data Loading
# =============================================================================

def load_patches(test_dir: str) -> dict:
    """
    Load all patches from <test_dir>/patches/.
    Returns {filename: BGR_image}.
    patch_0.png is guaranteed to exist and be readable.
    """
    patches_dir = Path(test_dir) / "patches"
    if not patches_dir.exists():
        raise FileNotFoundError(f"patches/ directory not found in: {test_dir}")

    imgs = {}
    bad  = []
    for p in sorted(patches_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            bad.append(p.name)
        else:
            imgs[p.name] = img

    print(f"[DATA] Loaded {len(imgs)} patches  ({patches_dir})")
    if bad:
        print(f"[WARN] Unreadable/corrupt patches (skipped): {bad}")
    if "patch_0.png" not in imgs:
        raise RuntimeError("patch_0.png is missing or unreadable – cannot proceed.")
    return imgs


def load_questions(test_dir: str) -> pd.DataFrame:
    """
    Load test.csv from test_dir.
    Columns expected: question_id, question, option_1, option_2, option_3, option_4
    """
    csv_path = Path(test_dir) / "test.csv"
    if not csv_path.exists():
        print(f"[WARN] test.csv not found in {test_dir}. Returning empty frame.")
        return pd.DataFrame(columns=[
            "question_id", "question",
            "option_1", "option_2", "option_3", "option_4",
        ])
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} question(s)  columns={list(df.columns)}")
    return df


# =============================================================================
# MODULE B – Preprocessing
# =============================================================================

def preprocess(img: np.ndarray) -> np.ndarray:
    """CLAHE + Gaussian denoise. Returns BGR."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(img, (3, 3), 0)


# =============================================================================
# MODULE C – Patch Grid Reconstruction
#
# Algorithm:
#   1. patch_0 is the TOP-LEFT anchor at (row=0, col=0, rotation=0). Fixed.
#   2. Determine best (rows, cols) grid for N patches (allow 0-2 missing).
#   3. Greedy left-to-right, top-to-bottom cell filling:
#      For each empty cell, test every unplaced patch at all 4 rotations.
#      Score = normalised cross-correlation (NCC) of the touching pixel strips
#              from the LEFT and TOP neighbours (averaged when both exist).
#   4. Place the highest-scoring (patch, rotation) pair.
#   5. Compose the final canvas.
# =============================================================================

def _edge_ncc(A: np.ndarray, B: np.ndarray, side: str) -> float:
    """
    Normalised cross-correlation of the touching edge strips.
    side='rl': right edge of A vs left edge of B
    side='bt': bottom edge of A vs top edge of B
    """
    w = CFG["EDGE_STRIP_W"]
    if side == "rl":
        ea = A[:, -w:, :].astype(np.float32)
        eb = B[:, :w,  :].astype(np.float32)
    else:  # bt
        ea = A[-w:, :, :].astype(np.float32)
        eb = B[:w,  :, :].astype(np.float32)
    ea = ea.flatten() - ea.mean()
    eb = eb.flatten() - eb.mean()
    nrm = np.linalg.norm(ea) * np.linalg.norm(eb)
    return float(np.dot(ea, eb) / nrm) if nrm > 1.0 else 0.0


def _cell_score(grid: dict, r: int, c: int, candidate: np.ndarray) -> float:
    """Average NCC against placed left and top neighbours."""
    scores, cnt = 0.0, 0
    if (r, c - 1) in grid:
        scores += _edge_ncc(grid[(r, c - 1)][0], candidate, "rl")
        cnt += 1
    if (r - 1, c) in grid:
        scores += _edge_ncc(grid[(r - 1, c)][0], candidate, "bt")
        cnt += 1
    return scores / cnt if cnt > 0 else 0.0


def _grid_dims(n: int) -> tuple:
    """
    Return (rows, cols) for the best grid fitting n patches.
    Prefers near-square grids; allows up to 2 missing cells.
    """
    best = None
    best_key = (999, 999)
    for cols in range(1, CFG["MAX_GRID_COLS"] + 1):
        for rows in range(1, CFG["MAX_GRID_ROWS"] + 1):
            total   = rows * cols
            missing = total - n
            if 0 <= missing <= 2:
                key = (missing, abs(rows - cols))
                if key < best_key:
                    best_key = key
                    best = (rows, cols)
    return best if best else (1, n)


def reconstruct_map(imgs_raw: dict) -> np.ndarray:
    """
    Reconstruct the full geospatial map from all loaded patches.
    patch_0.png is the fixed top-left anchor.
    Returns mosaic as BGR image.
    """
    # Preprocess
    imgs = {nm: preprocess(img) for nm, img in imgs_raw.items()}

    n = len(imgs)
    patch_h, patch_w = imgs["patch_0.png"].shape[:2]
    rows, cols = _grid_dims(n)
    print(f"[STITCH] {n} patches  each={patch_w}x{patch_h}  grid={rows}x{cols}")

    # grid: {(r,c): (rendered_img, orig_name, rotation_k)}
    grid    = {(0, 0): (imgs["patch_0.png"], "patch_0.png", 0)}
    placed  = {"patch_0.png"}
    unplaced = [nm for nm in imgs if nm != "patch_0.png"]

    for r in range(rows):
        for c in range(cols):
            if (r, c) in grid or not unplaced:
                continue

            best_sc  = -999.0
            best_nm  = None
            best_k   = 0
            best_rot = None

            for nm in unplaced:
                base = imgs[nm]
                for k in range(4):
                    rot = np.rot90(base, k)
                    sc  = _cell_score(grid, r, c, rot)
                    if sc > best_sc:
                        best_sc  = sc
                        best_nm  = nm
                        best_k   = k
                        best_rot = rot

            if best_nm is not None:
                grid[(r, c)] = (best_rot, best_nm, best_k)
                placed.add(best_nm)
                unplaced.remove(best_nm)

    # Compose canvas
    max_r = max(r for r, c in grid) + 1
    max_c = max(c for r, c in grid) + 1
    canvas = np.zeros((max_r * patch_h, max_c * patch_w, 3), dtype=np.uint8)
    for (r, c), (img, *_) in grid.items():
        canvas[r * patch_h:(r + 1) * patch_h,
               c * patch_w:(c + 1) * patch_w] = img

    print(f"[STITCH] Mosaic: {canvas.shape[1]}x{canvas.shape[0]}px  "
          f"({max_r}x{max_c} grid, {len(grid)}/{rows*cols} cells filled)")
    return canvas


# =============================================================================
# MODULE D – Scene Feature Extraction
# =============================================================================

def _gi(bgr):   # green index
    b, g, r = (bgr[:, :, i].astype(float) for i in range(3))
    return g / (b + g + r + 1e-6)

def _bd(bgr):   # blue dominance
    b, g, r = (bgr[:, :, i].astype(float) for i in range(3))
    return b / (r + g + 1e-6)


def extract_features(mosaic: np.ndarray) -> dict:
    """
    Extract interpretable scene features from the reconstructed map.
    Covers: colour, vegetation, water, urban, bare, texture, entropy,
    4-quadrant stats, 9-region (3x3) stats.
    """
    feat = {}
    h, w = mosaic.shape[:2]
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(mosaic, cv2.COLOR_BGR2HSV)

    # Global colour
    for i, nm in enumerate(["B", "G", "R"]):
        feat[f"mean_{nm}"] = float(mosaic[:, :, i].mean())
        feat[f"std_{nm}"]  = float(mosaic[:, :, i].std())
    feat["mean_H"] = float(hsv[:, :, 0].mean())
    feat["mean_S"] = float(hsv[:, :, 1].mean())
    feat["mean_V"] = float(hsv[:, :, 2].mean())

    # Land-cover proxies
    gi = _gi(mosaic)
    bd = _bd(mosaic)
    feat["veg_mean"]   = float(gi.mean())
    feat["veg_pct"]    = float((gi > 0.38).mean())
    feat["water_mean"] = float(bd.mean())
    feat["water_pct"]  = float((bd > 1.15).mean())
    feat["bare_pct"]   = float(((gi < 0.33) & (bd < 1.1)).mean())
    feat["urban_pct"]  = float(((gi < 0.33) & (bd < 1.1) &
                                 (mosaic.mean(axis=2) > 100)).mean())

    # Urban / edge metrics
    edges = cv2.Canny(gray, 50, 150)
    feat["edge_density"] = float(edges.mean() / 255.0)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                             minLineLength=30, maxLineGap=10)
    feat["line_count"] = 0 if lines is None else len(lines)
    corners = cv2.goodFeaturesToTrack(gray, 2000, 0.01, 5)
    feat["corner_density"] = 0.0 if corners is None else \
        len(corners) / (h * w + 1e-6)

    # Texture – LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    feat["lbp_mean"] = float(lbp.mean())
    feat["lbp_std"]  = float(lbp.std())

    # Texture – GLCM
    g256 = cv2.resize(gray, (256, 256))
    glcm = graycomatrix(g256, [1], [0], levels=256, symmetric=True, normed=True)
    feat["glcm_contrast"]    = float(graycoprops(glcm, "contrast")[0, 0])
    feat["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])
    feat["glcm_energy"]      = float(graycoprops(glcm, "energy")[0, 0])

    # Entropy
    hist, _ = np.histogram(gray.flatten(), 256, (0, 256), density=True)
    feat["entropy"] = float(scipy_entropy(hist + 1e-12))

    # 4-quadrant features
    qh, qw = h // 2, w // 2
    for qn, (rs, re, cs, ce) in {
        "TL": (0, qh, 0, qw), "TR": (0, qh, qw, w),
        "BL": (qh, h, 0, qw), "BR": (qh, h, qw, w),
    }.items():
        qi = mosaic[rs:re, cs:ce]
        qg = cv2.cvtColor(qi, cv2.COLOR_BGR2GRAY)
        feat[f"veg_{qn}"]    = float(_gi(qi).mean())
        feat[f"water_{qn}"]  = float(_bd(qi).mean())
        feat[f"edge_{qn}"]   = float(cv2.Canny(qg, 50, 150).mean() / 255.0)
        feat[f"bright_{qn}"] = float(qi.mean())

    # 9-region (3x3) features
    rh, rw = h // 3, w // 3
    regions = {
        "top_left":    (0,   rh,   0,   rw),
        "top_center":  (0,   rh,   rw,  2*rw),
        "top_right":   (0,   rh,   2*rw, w),
        "mid_left":    (rh,  2*rh, 0,   rw),
        "center":      (rh,  2*rh, rw,  2*rw),
        "mid_right":   (rh,  2*rh, 2*rw, w),
        "bot_left":    (2*rh, h,   0,   rw),
        "bot_center":  (2*rh, h,   rw,  2*rw),
        "bot_right":   (2*rh, h,   2*rw, w),
    }
    for rname, (rs, re, cs, ce) in regions.items():
        qi = mosaic[rs:re, cs:ce]
        qg = cv2.cvtColor(qi, cv2.COLOR_BGR2GRAY)
        feat[f"veg_{rname}"]    = float(_gi(qi).mean())
        feat[f"water_{rname}"]  = float(_bd(qi).mean())
        feat[f"edge_{rname}"]   = float(cv2.Canny(qg, 50, 150).mean() / 255.0)
        feat[f"bright_{rname}"] = float(qi.mean())

    return feat


def segment_scene(mosaic: np.ndarray) -> tuple:
    """K-Means pixel segmentation. Returns (label_map, cluster_centers)."""
    h, w = mosaic.shape[:2]
    b = mosaic[:, :, 0].flatten().astype(float) / 255
    g = mosaic[:, :, 1].flatten().astype(float) / 255
    r = mosaic[:, :, 2].flatten().astype(float) / 255
    gi = g / (b + g + r + 1e-6)
    bd = b / (r + g + 1e-6)
    X  = np.column_stack([b, g, r, gi, bd])
    km = KMeans(n_clusters=CFG["SEG_CLUSTERS"], random_state=42, n_init=5)
    labels = km.fit_predict(X).reshape(h, w)
    return labels, km.cluster_centers_


def classify_segments(labels: np.ndarray, centers: np.ndarray) -> dict:
    """Assign semantic tag to each K-Means cluster."""
    tags = {}
    for k, c in enumerate(centers):
        b, g, r, gi, bd = c
        if gi > 0.38:           tag = "vegetation"
        elif bd > 1.15:         tag = "water"
        elif (b + g + r) > 1.5: tag = "urban"
        else:                   tag = "bare"
        tags[k] = {"tag": tag, "area_pct": float((labels == k).mean()),
                   "center": c}
    return tags


# =============================================================================
# MODULE E – Question Understanding
# =============================================================================

Q_PATTERNS = {
    "counting"  : r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b",
    "largest"   : r"\blargest\b|\bbiggest\b|\bmost area\b|\bdominant\b|\bmaximum\b|\bpredominant\b",
    "smallest"  : r"\bsmallest\b|\bleast\b|\bminimum\b|\bfewest\b",
    "vegetation": r"\bveget\w*\b|\bgreen\b|\bforest\b|\btree\b|\bfarm\b|\bcrop\b|\bndvi\b|\bgrass\b|\bplant\b",
    "water"     : r"\bwater\b|\briver\b|\blake\b|\bocean\b|\bsea\b|\bpond\b|\bflow\b|\bstream\b",
    "urban"     : r"\burban\b|\bcity\b|\bbuilding\b|\broad\b|\bsettlement\b|\binfrastructure\b",
    "bare"      : r"\bbare\b|\bsoil\b|\bdesert\b|\bopen area\b",
    "position"  : r"\bquadrant\b|\bnorth\b|\bsouth\b|\beast\b|\bwest\b|\bdirection\b|\bleft\b|\bright\b|\btop\b|\bbottom\b|\bcorner\b|\bcenter\b|\bmiddle\b",
    "texture"   : r"\btexture\b|\brough\b|\bsmooth\b|\bhomogeneous\b|\bpattern\b|\buniform\b",
    "brightness": r"\bbright\b|\bdark\b|\bluminan\w*\b|\bintensity\b",
    "colour"    : r"\bcolou?r\b|\bwhite\b|\bbrown\b|\bred\b|\byellow\b",
}

QUAD_MAP = {
    "top left"    : "TL", "upper left"  : "TL", "north west" : "TL",
    "northwest"   : "TL", "top-left"    : "TL",
    "top right"   : "TR", "upper right" : "TR", "north east" : "TR",
    "northeast"   : "TR", "top-right"   : "TR",
    "bottom left" : "BL", "lower left"  : "BL", "south west" : "BL",
    "southwest"   : "BL", "bottom-left" : "BL",
    "bottom right": "BR", "lower right" : "BR", "south east" : "BR",
    "southeast"   : "BR", "bottom-right": "BR",
    "north"       : "TL", "south"       : "BL",
    "east"        : "TR", "west"        : "TL",
    "top"         : "TL", "bottom"      : "BL",
    "left"        : "TL", "right"       : "TR",
    "tl"          : "TL", "tr"          : "TR",
    "bl"          : "BL", "br"          : "BR",
}

REGION_MAP = {
    "top left"    : "top_left",   "upper left"   : "top_left",
    "top center"  : "top_center", "top middle"   : "top_center",
    "top right"   : "top_right",  "upper right"  : "top_right",
    "middle left" : "mid_left",   "center left"  : "mid_left",
    "center"      : "center",     "middle"       : "center",
    "middle right": "mid_right",  "center right" : "mid_right",
    "bottom left" : "bot_left",   "lower left"   : "bot_left",
    "bottom center":"bot_center", "bottom middle": "bot_center",
    "bottom right": "bot_right",  "lower right"  : "bot_right",
}

LANDCOVER_KWS = {
    "vegetation": "veg_mean",  "vegetat"   : "veg_mean",
    "green"     : "veg_mean",  "forest"    : "veg_mean",
    "tree"      : "veg_mean",  "crop"      : "veg_mean",
    "farm"      : "veg_mean",  "grass"     : "veg_mean",
    "plant"     : "veg_mean",
    "water"     : "water_mean","river"     : "water_mean",
    "lake"      : "water_mean","ocean"     : "water_mean",
    "sea"       : "water_mean","pond"      : "water_mean",
    "stream"    : "water_mean",
    "urban"     : "edge_density","city"    : "edge_density",
    "building"  : "edge_density","road"   : "edge_density",
    "settlement": "edge_density",
    "bare"      : "bare_pct",  "soil"      : "bare_pct",
    "desert"    : "bare_pct",
}


def detect_q_types(q: str) -> list:
    q_l = q.lower()
    matched = [qt for qt, pat in Q_PATTERNS.items()
               if re.search(pat, q_l)]
    return matched if matched else ["largest"]


def parse_question(row: pd.Series) -> dict:
    q_id = str(row.get("question_id", row.get("id", "")))
    q    = str(row.get("question", ""))
    opts = {i: str(row.get(f"option_{i}", "")) for i in range(1, 5)}
    return {
        "question_id": q_id,
        "question"   : q,
        "options"    : opts,
        "q_types"    : detect_q_types(q),
    }


def _quad_for(text: str):
    tl = text.lower()
    for phrase, qid in QUAD_MAP.items():
        if phrase in tl:
            return qid
    return None

def _region_for(text: str):
    tl = text.lower()
    for phrase, rn in REGION_MAP.items():
        if phrase in tl:
            return rn
    return None

def _cls_total(seg_tags: dict, cls: str) -> float:
    return sum(v["area_pct"] for v in seg_tags.values() if v["tag"] == cls)

def _dominant_cls(seg_tags: dict) -> str:
    if not seg_tags:
        return "unknown"
    return max(seg_tags.values(), key=lambda x: x["area_pct"])["tag"]


# =============================================================================
# MODULE F – MCQ Scoring
# =============================================================================

def score_options(pq: dict, feat: dict, seg_tags: dict) -> dict:
    """
    Score each option key (1..4). Returns {1: float, 2: float, 3: float, 4: float}.
    """
    opts    = pq["options"]
    q_types = pq["q_types"]
    q_l     = pq["question"].lower()
    scores  = {k: 0.0 for k in opts}

    for qt in q_types:

        if qt == "vegetation":
            for k, opt in opts.items():
                qid = _quad_for(opt)
                rn  = _region_for(opt)
                if qid:
                    scores[k] += feat.get(f"veg_{qid}", 0) * 2
                elif rn:
                    scores[k] += feat.get(f"veg_{rn}", 0) * 2
                elif any(kw in opt.lower() for kw in
                         ["veget","green","forest","tree","crop","farm","grass","plant"]):
                    scores[k] += _cls_total(seg_tags, "vegetation")

        elif qt == "water":
            for k, opt in opts.items():
                qid = _quad_for(opt)
                rn  = _region_for(opt)
                if qid:
                    scores[k] += feat.get(f"water_{qid}", 0) * 2
                elif rn:
                    scores[k] += feat.get(f"water_{rn}", 0) * 2
                elif any(kw in opt.lower() for kw in
                         ["water","river","lake","ocean","sea","pond","stream"]):
                    scores[k] += _cls_total(seg_tags, "water")

        elif qt == "urban":
            for k, opt in opts.items():
                qid = _quad_for(opt)
                rn  = _region_for(opt)
                if qid:
                    scores[k] += feat.get(f"edge_{qid}", 0) * 2
                elif rn:
                    scores[k] += feat.get(f"edge_{rn}", 0) * 2
                elif any(kw in opt.lower() for kw in
                         ["urban","city","building","road","settlement"]):
                    scores[k] += _cls_total(seg_tags, "urban")

        elif qt == "bare":
            for k, opt in opts.items():
                if any(kw in opt.lower() for kw in ["bare","soil","desert","open"]):
                    scores[k] += _cls_total(seg_tags, "bare")

        elif qt == "largest":
            dom = _dominant_cls(seg_tags)
            for k, opt in opts.items():
                opt_l = opt.lower()
                for cls in ["vegetation","water","urban","bare"]:
                    if cls in opt_l and cls == dom:
                        scores[k] += 2.0
                qid = _quad_for(opt)
                if qid:
                    fk = ("veg" if dom == "vegetation"
                          else "water" if dom == "water"
                          else "edge")
                    qv = {q: feat.get(f"{fk}_{q}", 0)
                          for q in ["TL","TR","BL","BR"]}
                    if qid == max(qv, key=qv.get):
                        scores[k] += 1.5

        elif qt == "smallest":
            if seg_tags:
                ct = {}
                for v in seg_tags.values():
                    ct[v["tag"]] = ct.get(v["tag"], 0) + v["area_pct"]
                least = min(ct, key=ct.get)
                for k, opt in opts.items():
                    if least in opt.lower():
                        scores[k] += 2.0

        elif qt == "counting":
            n_reg = len(seg_tags) if seg_tags else CFG["SEG_CLUSTERS"]
            for k, opt in opts.items():
                nums = re.findall(r"\d+", opt)
                if nums:
                    scores[k] += 1.0 / (1.0 + abs(int(nums[0]) - n_reg))

        elif qt == "position":
            rel = "veg"
            if any(t in q_l for t in ["water","river","lake","ocean","sea"]):
                rel = "water"
            elif any(t in q_l for t in ["urban","city","building","road"]):
                rel = "edge"
            elif any(t in q_l for t in ["bright","light"]):
                rel = "bright"
            qv = {q: feat.get(f"{rel}_{q}", 0) for q in ["TL","TR","BL","BR"]}
            best_q = max(qv, key=qv.get)
            labels_map = {
                "TL": ["top left","upper left","northwest","north west","tl","north","top","left"],
                "TR": ["top right","upper right","northeast","north east","tr","east","right"],
                "BL": ["bottom left","lower left","southwest","south west","bl","south","bottom"],
                "BR": ["bottom right","lower right","southeast","south east","br"],
            }
            for k, opt in opts.items():
                opt_l = opt.lower()
                if any(lbl in opt_l for lbl in labels_map.get(best_q, [])):
                    scores[k] += 2.0

        elif qt == "texture":
            for k, opt in opts.items():
                opt_l = opt.lower()
                if any(t in opt_l for t in ["rough","complex","heterogeneous","high","varied"]):
                    scores[k] += feat.get("glcm_contrast", 0) / 200.0
                    scores[k] += feat.get("lbp_std", 0) / 10.0
                elif any(t in opt_l for t in ["smooth","homogeneous","uniform","low"]):
                    scores[k] += feat.get("glcm_homogeneity", 0)
                    scores[k] += feat.get("glcm_energy", 0)

        elif qt in ("brightness", "colour"):
            for k, opt in opts.items():
                qid = _quad_for(opt)
                opt_l = opt.lower()
                if qid:
                    scores[k] += feat.get(f"bright_{qid}", 0) / 255.0
                elif any(t in opt_l for t in ["bright","light"]):
                    scores[k] += feat.get("mean_V", 128) / 255.0
                elif "dark" in opt_l:
                    scores[k] += 1.0 - feat.get("mean_V", 128) / 255.0

    # Soft keyword fallback (always applied)
    for k, opt in opts.items():
        opt_l = opt.lower()
        for kw, fkey in LANDCOVER_KWS.items():
            if kw in opt_l:
                scores[k] += feat.get(fkey, 0) * 0.3

    return scores


# =============================================================================
# MODULE G – Ensemble & Confidence-Aware Final Prediction
# =============================================================================

def _rule_scores(pq: dict, feat: dict) -> dict:
    s = {k: 0.0 for k in pq["options"]}
    for k, opt in pq["options"].items():
        for kw, fk in LANDCOVER_KWS.items():
            if kw in opt.lower():
                s[k] += feat.get(fk, 0)
    return s


def _quad_scores(pq: dict, feat: dict) -> dict:
    s = {k: 0.0 for k in pq["options"]}
    for k, opt in pq["options"].items():
        qid = _quad_for(opt)
        if qid:
            s[k] += (feat.get(f"veg_{qid}", 0) +
                     feat.get(f"water_{qid}", 0) * 0.5 +
                     feat.get(f"edge_{qid}", 0) * 0.3)
    return s


def _seg_scores(pq: dict, seg_tags: dict) -> dict:
    s = {k: 0.0 for k in pq["options"]}
    ct = {}
    for v in seg_tags.values():
        ct[v["tag"]] = ct.get(v["tag"], 0) + v["area_pct"]
    for k, opt in pq["options"].items():
        for cls, tot in ct.items():
            if cls in opt.lower():
                s[k] += tot
    return s


def predict(pq: dict, feat: dict, seg_tags: dict) -> int:
    """
    Ensemble weighted vote → confidence check → integer in {1,2,3,4,5}.
    Returns 5 (unanswered, 0 penalty) when confidence gap is too small.
    """
    s1 = score_options(pq, feat, seg_tags)
    s2 = _rule_scores(pq, feat)
    s3 = _quad_scores(pq, feat)
    s4 = _seg_scores(pq, seg_tags)

    combined = {}
    for sd, w in zip([s1, s2, s3, s4], [2.0, 1.0, 1.5, 1.5]):
        total = sum(sd.values()) + 1e-9
        for k, v in sd.items():
            combined[k] = combined.get(k, 0.0) + w * (v / total)

    total_c = sum(combined.values()) + 1e-9
    norm    = {k: v / total_c for k, v in combined.items()}

    sorted_vals = sorted(norm.values(), reverse=True)
    best_k = max(norm, key=norm.get)
    gap    = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 1.0

    # If no meaningful signal, abstain (0 penalty rather than -0.25)
    if gap < CFG["CONF_THRESHOLD"]:
        return 5

    return int(best_k)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNR638 Project 1 – Geospatial Stitching & MCQ Inference")
    parser.add_argument(
        "--test_dir", required=True,
        help="Absolute path to test directory (contains patches/ and test.csv)")
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*65}")
    print(f" GNR638 Project 1 – Inference")
    print(f" test_dir = {args.test_dir}")
    print(f"{'='*65}\n")

    # ── Load ──────────────────────────────────────────────────────
    imgs_raw  = load_patches(args.test_dir)
    questions = load_questions(args.test_dir)

    # ── Reconstruct map ───────────────────────────────────────────
    print("\n[STITCH] Reconstructing map...")
    mosaic = reconstruct_map(imgs_raw)

    # ── Extract features ──────────────────────────────────────────
    print("\n[FEAT] Extracting features...")
    feat            = extract_features(mosaic)
    labels, centers = segment_scene(mosaic)
    seg_tags        = classify_segments(labels, centers)
    seg_summary     = {v["tag"]: f"{v['area_pct']:.1%}"
                       for v in seg_tags.values()}
    print(f"[FEAT] Segments: {seg_summary}")

    # ── Answer questions ──────────────────────────────────────────
    print(f"\n[MCQ] {len(questions)} question(s)\n")
    rows = []
    for _, row in questions.iterrows():
        pq     = parse_question(row)
        answer = predict(pq, feat, seg_tags)
        q_id   = pq["question_id"]

        # Safety: force to valid set
        if answer not in VALID_OPTIONS:
            print(f"  [SAFETY] Forced invalid {answer} → 5 for {q_id}")
            answer = 5

        print(f"  {q_id} | Q: {pq['question'][:70]}")
        for ok, ov in pq["options"].items():
            print(f"         |  option_{ok}: {ov}"
                  + ("  <-- PREDICTED" if ok == answer else ""))
        print(f"         | types={pq['q_types']}  answer={answer}\n")

        rows.append({"id": q_id, "question_num": q_id, "option": answer})

    # ── Write submission.csv to CWD ───────────────────────────────
    sub_df   = pd.DataFrame(rows, columns=["id", "question_num", "option"])

    # Final safety check
    bad = sub_df[~sub_df["option"].isin(VALID_OPTIONS)]
    if not bad.empty:
        print(f"[SAFETY] Overriding {len(bad)} invalid option(s) → 5")
        sub_df.loc[~sub_df["option"].isin(VALID_OPTIONS), "option"] = 5

    out_path = Path("submission.csv")          # CWD, NOT test_dir
    sub_df.to_csv(out_path, index=False)

    print(f"\n{'='*65}")
    print(f" Output   : {out_path.resolve()}")
    print(f" Runtime  : {time.time()-t0:.1f}s")
    print(f"{'='*65}")
    print(sub_df.to_string(index=False))


if __name__ == "__main__":
    main()
