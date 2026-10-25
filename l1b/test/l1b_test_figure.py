"""Plot original L1B equalization profiles for all four VNIR bands."""
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


BASE_DIR = Path(
    r"C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri"
    r"\EODP\EODP_TER_2021\EODP-TS-L1B"
)
EQ_DIR = BASE_DIR / "output_test_mio"
NO_EQ_DIR = BASE_DIR / "output_dia2_noecualiz"
TRUTH_DIR = BASE_DIR / "input"
VARIABLE = "toa"
ALT_INDEX = 50
BANDS = tuple(f"VNIR-{index}" for index in range(4))


def confirm_run_labels(directory: Path, expected_equalization: bool) -> None:
    """Check the latest logged run before assigning equalization labels."""
    log_file = directory / "L1B.log"
    text = log_file.read_text(encoding="utf-8")
    marker = "Start of the L1B Processing Module"
    if marker not in text:
        raise ValueError(f"Cannot confirm run labels: no L1B run found in {log_file}")
    latest = text.rsplit(marker, 1)[1]
    if "End of the L1B Module!" not in latest:
        raise ValueError(f"Cannot confirm run labels: latest run is incomplete in {log_file}")
    for band in BANDS:
        start, end = f"Start of BAND {band}", f"End of BAND {band}"
        if start not in latest or end not in latest:
            raise ValueError(f"Cannot confirm run labels for {band} in {log_file}")
        section = latest.split(start, 1)[1].split(end, 1)[0]
        equalized = "EODP-ALG-L1B-1010: Radiometric Correction (equalization)" in section
        restored = "EODP-ALG-L1B-1020: Absolute radiometric gain application" in section
        if equalized != expected_equalization or not restored:
            raise ValueError(f"Run evidence contradicts the requested label for {band} in {log_file}")
    label = "with equalization" if expected_equalization else "without equalization"
    print(f"Confirmed {label} for all four bands from {log_file}")


def read_toa_profile(file_path, variable="toa", alt_index=50):
    """Read the selected ALT row as a float64 ACT profile."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with xr.open_dataset(file_path) as dataset:
        if variable not in dataset.variables:
            raise ValueError(
                f"Variable '{variable}' not found in {file_path}. "
                f"Available variables: {list(dataset.variables)}"
            )
        data = dataset[variable]
        print(f"File: {file_path}")
        print(f"Variable: {variable}; dimensions: {data.dims}; shape: {data.shape}")
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D TOA variable in {file_path}, found {data.ndim} dimensions")
        if not 0 <= alt_index < data.shape[0]:
            raise IndexError(f"ALT_INDEX={alt_index} is outside the first dimension of size {data.shape[0]}")
        profile = np.asarray(data.isel({data.dims[0]: alt_index}).values, dtype=np.float64)
        print(f"Selected profile: ALT row {alt_index}; ACT pixels: {profile.size}")
        return profile


def compute_metrics(signal, truth):
    """Return RMSE, MAE and the number of finite paired samples."""
    valid = np.isfinite(signal) & np.isfinite(truth)
    if not np.any(valid):
        raise ValueError("No finite paired samples are available for error metrics")
    error = signal[valid] - truth[valid]
    return float(np.sqrt(np.mean(error ** 2))), float(np.mean(np.abs(error))), int(valid.sum())


def main() -> None:
    # The processing code logs L1B-1010 only inside the do_equalization branch,
    # before restitution (L1B-1020) and writing the final l1b_toa product.
    # Check both logs before reading data or generating any figures.
    try:
        confirm_run_labels(EQ_DIR, expected_equalization=True)
        confirm_run_labels(NO_EQ_DIR, expected_equalization=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot confirm equalization labels; no figures generated. {error}") from error

    metrics = []
    for band in BANDS:
        print()
        print("=" * 70)
        print(f"BAND {band}: ALT={ALT_INDEX}")
        toa_eq = read_toa_profile(EQ_DIR / f"l1b_toa_{band}.nc", VARIABLE, ALT_INDEX)
        toa_no_eq = read_toa_profile(NO_EQ_DIR / f"l1b_toa_{band}.nc", VARIABLE, ALT_INDEX)
        toa_truth = read_toa_profile(TRUTH_DIR / f"ism_toa_isrf_{band}.nc", VARIABLE, ALT_INDEX)
        if not (toa_eq.shape == toa_no_eq.shape == toa_truth.shape):
            raise ValueError(f"The three profiles do not have the same number of samples for {band}")
        rmse_eq, mae_eq, count_eq = compute_metrics(toa_eq, toa_truth)
        rmse_no_eq, mae_no_eq, count_no_eq = compute_metrics(toa_no_eq, toa_truth)
        metrics.append((band, rmse_eq, mae_eq, rmse_no_eq, mae_no_eq))
        print("ERRORS RELATIVE TO GROUND TRUTH [mW/m2/sr]")
        print(f"With equalization: RMSE={rmse_eq:.6f}; MAE={mae_eq:.6f}; valid samples={count_eq}")
        print(f"Without equalization: RMSE={rmse_no_eq:.6f}; MAE={mae_no_eq:.6f}; valid samples={count_no_eq}")

        act_pixel = np.arange(len(toa_truth))
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.plot(act_pixel, toa_eq, color="black", linewidth=1.5, label="TOA L1B with eq")
        ax.plot(act_pixel, toa_no_eq, color="red", linewidth=1.5, label="TOA L1B no eq")
        ax.plot(act_pixel, toa_truth, color="blue", linewidth=2, label="TOA after the ISRF")
        ax.set_title(f"Effect of the Equalization for {band}", fontsize=15)
        ax.set_xlabel("ACT pixel [-]", fontsize=12)
        ax.set_ylabel("TOA [mW/m²/sr]", fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left")
        fig.tight_layout()
        output_figure = BASE_DIR / f"Figure_8_3_Equalization_{band}.png"
        fig.savefig(output_figure, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved at 300 dpi: {output_figure}")

    print()
    print("FINAL METRICS RELATIVE TO GROUND TRUTH [mW/m2/sr]")
    print(f"{'Band':<10} {'RMSE with eq':>15} {'MAE with eq':>15} {'RMSE no eq':>15} {'MAE no eq':>15}")
    for band, rmse_eq, mae_eq, rmse_no_eq, mae_no_eq in metrics:
        print(f"{band:<10} {rmse_eq:15.6f} {mae_eq:15.6f} {rmse_no_eq:15.6f} {mae_no_eq:15.6f}")


if __name__ == "__main__":
    main()
