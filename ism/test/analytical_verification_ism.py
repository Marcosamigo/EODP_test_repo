"""Verify original ISM conversions from existing NetCDF pixels; write no outputs."""
from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
from netCDF4 import Dataset


EODP_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = EODP_DIR / "EODP_TER_2021" / "EODP-TS-ISM" / "output_test"
ORIGINAL_ROOT = EODP_DIR / "eodp_students-master" / "eodp_students-master"
CONFIG_FILE = ORIGINAL_ROOT / "config" / "ismConfig.py"
CONSTANTS_FILE = ORIGINAL_ROOT / "auxiliary" / "constants.py"
PIXEL = (50, 75)  # Zero-based [ALT, ACT].
BANDS = range(4)
# Intermediate NetCDF values are stored as float32, unlike internal calculations.
ELECTRON_ATOL = 1e-3
ELECTRON_RTOL = 1e-6


def load_original_parameters():
    """Load the original student configuration, never the repository E2E config."""
    config = runpy.run_path(str(CONFIG_FILE))["ismConfig"]()
    constants = runpy.run_path(str(CONSTANTS_FILE))["constants"]()
    required = {"pix_size": 30e-6, "t_int": 0.00672, "D": 0.150, "f": 0.5262}
    for name, expected in required.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(
                f"Original configuration required: {name}={expected}, found {actual}. "
                "Do not use the modified E2E configuration."
            )
    if len(config.wv) != 4:
        raise ValueError("The original configuration must contain four VNIR wavelengths.")
    return config, constants


def read_pixel(product: str, band: int) -> float:
    """Read only the selected pixel from an existing product, in read-only mode."""
    path = OUTPUT_DIR / f"{product}_VNIR-{band}.nc"
    with Dataset(path, mode="r") as dataset:
        if "toa" not in dataset.variables:
            raise ValueError(f"Missing 'toa' variable in {path.name}")
        variable = dataset.variables["toa"]
        if variable.ndim != 2 or any(i >= n for i, n in zip(PIXEL, variable.shape)):
            raise ValueError(f"Pixel {PIXEL} is outside the 2D 'toa' variable in {path.name}")
        value = variable[PIXEL]
        if np.ma.is_masked(value) or not np.isfinite(value):
            raise ValueError(f"Invalid pixel {PIXEL} in {path.name}")
        return float(value)


def print_conversion_table(rows: list[tuple[str, str, str, str, str]]) -> None:
    """Print a five-column console table without truncating values or factors."""
    headers = ("Stage", "Units", "Conversion factor", "Analytical value", "NetCDF value")
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(5)]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print("| " + " | ".join(value.ljust(width) for value, width in zip(headers, widths)) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |")
    print(separator)


def verify_band(band: int, config, constants) -> bool:
    """Tabulate a radiance-based chain and check independent stored stages."""
    radiance = read_pixel("ism_toa_isrf", band)
    optical = read_pixel("ism_toa_optical", band)
    stored_electrons = read_pixel("ism_toa_e", band)
    detection = read_pixel("ism_toa_detection", band)
    stored_dn = read_pixel("ism_toa", band)
    wavelength = float(config.wv[band])

    factor = config.Tr * (np.pi / 4) * (config.D / config.f) ** 2
    pre_mtf_irradiance = radiance * factor
    pixel_area = config.pix_size * config.pix_size
    incident_energy = optical * 1e-3 * pixel_area * config.t_int
    photon_energy = constants.h_planck * constants.speed_light / wavelength
    photons = incident_energy / photon_energy
    unclipped_electrons = photons * config.QE
    # The original implementation caps electrons after QE, despite the config's [ph] comment.
    electrons = min(unclipped_electrons, config.FWC)
    electron_difference = electrons - stored_electrons
    electrons_ok = bool(np.isclose(
        electrons, stored_electrons, atol=ELECTRON_ATOL, rtol=ELECTRON_RTOL
    ))

    voltage = detection * config.OCF * config.ADC_gain
    max_dn = 2 ** config.bit_depth - 1
    # Match original digitisation exactly: no subtraction of min_voltage.
    raw_dn = voltage / (config.max_voltage - config.min_voltage) * max_dn
    rounded_dn = float(np.round(raw_dn))  # NumPy rounds ties to even.
    dn = float(np.clip(rounded_dn, 0, max_dn))
    dn_difference = dn - stored_dn
    dn_ok = dn == stored_dn

    # This chain starts only from ISRF radiance. Do not restart it from optical
    # or detection NetCDF values: their MTF/noise effects are not modeled here.
    radiance_si = radiance * 1e-3
    chain_irradiance = radiance_si * factor
    photon_factor = pixel_area * config.t_int / photon_energy
    chain_photons = chain_irradiance * photon_factor
    chain_electrons_raw = chain_photons * config.QE
    chain_electrons = min(chain_electrons_raw, config.FWC)
    voltage_factor = config.OCF * config.ADC_gain
    chain_voltage = chain_electrons * voltage_factor
    dn_factor = max_dn / (config.max_voltage - config.min_voltage)
    chain_dn_raw = chain_voltage * dn_factor
    chain_dn_rounded = float(np.round(chain_dn_raw))
    chain_dn = float(np.clip(chain_dn_rounded, 0, max_dn))

    print(f"Wavelength: {wavelength * 1e9:.6f} nm")
    print("Analytical values form one chain from ISRF radiance, without MTF or detection effects.")
    print("NetCDF values show the actual stored stages; chain differences are not pass/fail tests.")
    rows = [
        ("Radiance after ISRF", "W/m2/sr", "x 1.000000000e-03 (mW -> W)",
         f"{radiance_si:.12e}", f"{radiance_si:.12e}"),
        ("Irradiance before MTF", "W/m2", f"x {factor:.12e}",
         f"{chain_irradiance:.12e}", "Not stored"),
        ("Irradiance after MTF", "W/m2", "Spatial filtering; no single factor",
         "Not modeled", f"{optical * 1e-3:.12e}"),
        ("Photons", "photons", f"x {photon_factor:.12e}",
         f"{chain_photons:.12e}", "Not stored"),
        ("Electrons after FWC", "e-", f"x {config.QE:.12e}; min(result, {config.FWC:g})",
         f"{chain_electrons:.12e}", f"{stored_electrons:.12e}"),
        ("Electrons after detection", "e-", "Detection effects; no single factor",
         "Not modeled", f"{detection:.12e}"),
        ("Voltage", "V", f"x {voltage_factor:.12e}",
         f"{chain_voltage:.12e}", "Not stored"),
        ("Digital number", "DN", f"x {dn_factor:.12e}; round; clip[0,{max_dn}]",
         f"{chain_dn:.0f}", f"{stored_dn:.0f}"),
    ]
    print_conversion_table(rows)
    print("Factors: Tr*pi/4*(D/f)^2; pixel_area*t_int/(h*c/wavelength); QE;")
    print("         OCF*ADC_gain; (2^bit_depth-1)/(max_voltage-min_voltage).")
    print("The Photons row uses irradiance BEFORE MTF, not the stored after-MTF row.")
    print("The Voltage row uses analytical FWC-limited electrons, not stored detection electrons.")
    print("MTF and detection rows provide context only; no analytical equivalent is assumed.")
    print(f"Chain electrons before FWC: {chain_electrons_raw:.12e}; "
          f"FWC applied: {chain_electrons_raw > config.FWC}")
    print(f"Chain DN before rounding: {chain_dn_raw:.12e}; rounded: {chain_dn_rounded:.0f}; "
          f"saturation applied: {chain_dn != chain_dn_rounded}")
    print("Rounding uses NumPy ties-to-even; the original ADC formula does not subtract min_voltage.")
    print()
    print("INDEPENDENT CHECKS (using stored stage inputs)")
    print(f"Optical -> electrons: photons={photons:.12e}; "
          f"analytical={electrons:.12e} e-; NetCDF={stored_electrons:.12e} e-")
    print(f"Difference (analytical - NetCDF): {electron_difference:.12e} e-; "
          f"{'PASSED' if electrons_ok else 'FAILED'}")
    print(f"Detection -> DN: voltage={voltage:.12e} V; raw DN={raw_dn:.12e}; "
          f"rounded DN={rounded_dn:.0f}")
    print(f"Analytical={dn:.0f} DN; NetCDF={stored_dn:.0f} DN; "
          f"difference={dn_difference:.12e} DN; {'PASSED' if dn_ok else 'FAILED'}")
    return electrons_ok and dn_ok


def run_analytical_verification() -> bool:
    """Report four bands without running the ISM or creating any data files."""
    print("ORIGINAL ISM ANALYTICAL VERIFICATION")
    print(f"Pixel: [{PIXEL[0]},{PIXEL[1]}] (zero-based ALT, ACT)")
    print(f"Input directory: {OUTPUT_DIR}")
    print(f"Original configuration: {CONFIG_FILE}")
    print(f"Physical constants: {CONSTANTS_FILE}")
    try:
        config, constants = load_original_parameters()
    except (OSError, KeyError, ValueError) as error:
        print(f"Configuration error: {error}")
        print("GLOBAL RESULT: FAILED (original parameters could not be loaded)")
        return False

    print(f"pix_size={config.pix_size} m; t_int={config.t_int} s; D={config.D} m; f={config.f} m")
    print(f"Tr={config.Tr}; QE={config.QE}; FWC={config.FWC} electrons")
    print(f"OCF={config.OCF} V/e-; ADC_gain={config.ADC_gain}; bit_depth={config.bit_depth}")
    print(f"Voltage range: [{config.min_voltage}, {config.max_voltage}] V")
    print(f"h={constants.h_planck} J s; c={constants.speed_light} m/s")
    print(f"Electron tolerance: atol={ELECTRON_ATOL} e-, rtol={ELECTRON_RTOL} (float32 storage)")
    print("DN comparison requires exact equality after rounding and saturation.")
    print("Radiance-to-irradiance conversion precedes spatial MTF filtering.")
    print("Only optical-to-electron and detection-to-DN conversions are validated here.")
    results = {}
    for band in BANDS:
        print("=" * 80)
        print(f"BAND VNIR-{band}")
        try:
            results[band] = "PASSED" if verify_band(band, config, constants) else "FAILED"
        except FileNotFoundError as error:
            print(f"Missing input: {error}")
            results[band] = "MISSING"
        except (OSError, ValueError, KeyError, IndexError) as error:
            print(f"Verification error: {error}")
            results[band] = "FAILED"
        print(f"BAND RESULT: {results[band]}")

    print("=" * 80)
    print("FINAL SUMMARY")
    print("Scope: deterministic conversions at pixel [50,75] in four VNIR bands; MTF is not validated.")
    for band, status in results.items():
        print(f"VNIR-{band}: {status}")
    print(f"Total bands: {len(results)}")
    for status in ("PASSED", "FAILED", "MISSING"):
        print(f"{status}: {sum(value == status for value in results.values())}")
    all_ok = all(value == "PASSED" for value in results.values())
    print(f"GLOBAL RESULT: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if run_analytical_verification() else 1)
