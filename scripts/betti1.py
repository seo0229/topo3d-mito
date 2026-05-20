import os
import numpy as np
import tifffile
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------ #
# Betti-1 Error (cube-wise via cripser)
# ------------------------------------------------------------------ #

def betti_1_cube(vol_cube):
    import cripser
    filtration = 1.0 - vol_cube.astype(np.float64)  # foreground → 0.0
    ph = cripser.computePH(filtration, maxdim=1)
    if len(ph) == 0:
        return 0
    return int(np.sum((ph[:, 0] == 1) & (ph[:, 1] == 0.0)))


def compute_betti_1_error(pred, gt, cube_size=64):
    """
    Cube-wise Betti-1 error: averages |β₁(pred) - β₁(gt)| over all
    non-overlapping cube_size³ subvolumes. Skips fully empty cubes.
    """
    Z, Y, X = pred.shape
    errors = []
    for z in range(0, Z, cube_size):
        for y in range(0, Y, cube_size):
            for x in range(0, X, cube_size):
                pc = pred[z:z+cube_size, y:y+cube_size, x:x+cube_size]
                gc =   gt[z:z+cube_size, y:y+cube_size, x:x+cube_size]
                if pc.sum() == 0 and gc.sum() == 0:
                    continue
                errors.append(abs(betti_1_cube(pc) - betti_1_cube(gc)))
    return float(np.mean(errors)) if errors else 0.0


# ------------------------------------------------------------------ #
# Per-model worker (called in a separate process)
# ------------------------------------------------------------------ #

def evaluate_model(model_name, pred_paths, gt_paths, output_dir, cube_size=64):
    """
    Loads each (pred, gt) pair, computes Betti-1 error, and writes
    results to output_dir/betti1_results.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "betti1_results.txt")

    pair_errors = []
    lines = []
    lines.append("=" * 60)
    lines.append(f"BETTI-1 ERROR RESULTS — {model_name}")
    lines.append("=" * 60)
    lines.append(f"Cube size : {cube_size}³")
    lines.append(f"Volumes   : {len(pred_paths)}")
    lines.append("")

    for i, (pred_path, gt_path) in enumerate(zip(pred_paths, gt_paths)):
        print(f"[{model_name}] Volume {i+1}/{len(pred_paths)} — loading...")

        pred = tifffile.imread(pred_path)
        gt   = tifffile.imread(gt_path)

        pred = (np.squeeze(pred).astype(np.float32) / 255 > 0.5).astype(np.uint8)
        gt   = (np.squeeze(gt).astype(np.float32)         > 0.5).astype(np.uint8)

        assert pred.ndim == 3 and gt.ndim == 3, (
            f"Expected 3D volumes, got pred={pred.shape}, gt={gt.shape}"
        )

        print(f"[{model_name}] Volume {i+1} — computing Betti-1 error "
              f"(shape {pred.shape}, cube_size={cube_size})...")

        error = compute_betti_1_error(pred, gt, cube_size=cube_size)
        pair_errors.append(error)

        print(f"[{model_name}] Volume {i+1} — Betti-1 error: {error:.4f}")

        lines.append(f"--- Pair {i+1} ---")
        lines.append(f"  Prediction  : {os.path.basename(pred_path)}")
        lines.append(f"  Ground Truth: {os.path.basename(gt_path)}")
        lines.append(f"  Betti-1 Error (mean over cubes): {error:.4f}")
        lines.append("")

    avg = np.mean(pair_errors)
    std = np.std(pair_errors)
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)
    for i, e in enumerate(pair_errors):
        lines.append(f"  Volume {i+1}: {e:.4f}")
    lines.append(f"  Average : {avg:.4f} ± {std:.4f}")

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[{model_name}] Done. Results saved to: {output_file}")
    return model_name, pair_errors


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# Ground truth files — shared across all models (one per volume)
gt_files = [
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw1.tif',
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw4.tif',
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw5.tif',
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw7.tif',
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw28.tif',
    '/data/elliott/baselines/pytorch_connectomics/datasets/MitoAICS/allfiles/3Dedited/raw29.tif',
]

# Each entry: model display name → (list of pred paths, output results dir)
# pred paths must correspond 1-to-1 with gt_files above
MODELS = {
    '3D-GTO': (
        [
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_000.tiff',
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_001.tiff',
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_002.tiff',
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_003.tiff',
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_004.tiff',
            '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/prediction_005.tiff',
        ],
        '/data/elliott/baselines/pytorch_connectomics/outputs/baseline/3D-GTO/test/results',
    )
}

CUBE_SIZE = 64

# ------------------------------------------------------------------ #
# Main — parallel execution across models
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    print("=" * 60)
    print("BETTI-1 EVALUATION — MULTI-MODEL PARALLEL")
    print("=" * 60)
    print(f"Models    : {list(MODELS.keys())}")
    print(f"Volumes   : {len(gt_files)}")
    print(f"Cube size : {CUBE_SIZE}³")
    print()

    futures = {}
    # One worker process per model
    with ProcessPoolExecutor(max_workers=len(MODELS)) as executor:
        for model_name, (pred_paths, output_dir) in MODELS.items():
            assert len(pred_paths) == len(gt_files), (
                f"Model '{model_name}' has {len(pred_paths)} predictions "
                f"but {len(gt_files)} GT files."
            )
            future = executor.submit(
                evaluate_model,
                model_name, pred_paths, gt_files, output_dir, CUBE_SIZE
            )
            futures[future] = model_name

    print("\n" + "=" * 60)
    print("ALL MODELS COMPLETE")
    print("=" * 60)
    for future in as_completed(futures):
        model_name, errors = future.result()
        avg = np.mean(errors)
        std = np.std(errors)
        print(f"  {model_name:<20} avg Betti-1 error: {avg:.4f} ± {std:.4f}")
