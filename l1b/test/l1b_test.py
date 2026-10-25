# CROSS VALIDATE L1B OUTPUTS EQUALIZED

# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH
# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc

#Comparar los resultados buenos (C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri\EODP\EODP_TER_2021\EODP-TS-L1B\output),
#con los resultados mios (C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri\EODP\EODP_TER_2021\EODP-TS-L1B\output_test_mio)

#Generar el script correspondiente

#Hacer el plot figura 8.3, plotear las 3 lineas
#-Datos equalizados
#-Datos NO equalizados
#-Linea verdadera (TRUTH)

#Explicar el plot, que son cada linea, por que la negra no es exactamente la linea azul

#Para cada test necesitamos el test report, github y los resultados, para el l1b, tenemos que entregar el crossvalidation, el plot y el comentario del plot

from pathlib import Path

import numpy as np
import xarray as xr


ABS_TOL = 1e-8
REL_TOL = 1e-6

REFERENCE_DIR = Path(
    r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-L1B\\output"
)

TEST_DIR = Path(
    r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-L1B\\output_test_mio"
)


def is_numeric_array(values: np.ndarray) -> bool:
    """Indica si un array tiene un tipo numerico comparable con tolerancias."""
    return np.issubdtype(values.dtype, np.number)


def max_relative_difference(reference: np.ndarray, test: np.ndarray) -> float:
    """Calcula la diferencia relativa maxima usando el valor de referencia."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)

    diff = np.abs(ref - tst)
    denominator = np.abs(ref)

    valid = np.isfinite(diff) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(valid):
        if np.allclose(ref, tst, atol=ABS_TOL, rtol=REL_TOL, equal_nan=True):
            return 0.0
        return float("inf")

    return float(np.max(diff[valid] / denominator[valid]))


def rmse(reference: np.ndarray, test: np.ndarray) -> float:
    """Calcula el RMSE ignorando posiciones donde ambos valores son NaN."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)

    both_nan = np.isnan(ref) & np.isnan(tst)
    diff = ref[~both_nan] - tst[~both_nan]

    if diff.size == 0:
        return 0.0

    return float(np.sqrt(np.nanmean(diff**2)))


def arrays_equal_non_numeric(reference: np.ndarray, test: np.ndarray) -> bool:
    """Compara arrays no numericos, aceptando NaN si existen en arrays object."""
    try:
        return bool(np.array_equal(reference, test, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(reference, test))


def compare_dimensions(reference_ds: xr.Dataset, test_ds: xr.Dataset) -> bool:
    """Compara las dimensiones globales de ambos datasets."""
    dimensions_ok = True

    reference_dims = dict(reference_ds.sizes)
    test_dims = dict(test_ds.sizes)

    only_reference = sorted(set(reference_dims) - set(test_dims))
    only_test = sorted(set(test_dims) - set(reference_dims))
    common_dims = sorted(set(reference_dims) & set(test_dims))

    if only_reference:
        dimensions_ok = False
        print(f"  Dimensions only in reference: {only_reference}")

    if only_test:
        dimensions_ok = False
        print(f"  Dimensions only in test: {only_test}")

    for dim in common_dims:
        ref_size = reference_dims[dim]
        test_size = test_dims[dim]
        if ref_size == test_size:
            print(f"  Dimension {dim}: OK ({ref_size})")
        else:
            dimensions_ok = False
            print(f"  Dimension {dim}: DIFFERENT reference={ref_size}, test={test_size}")

    return dimensions_ok


def compare_variable(variable_name: str, reference_var: xr.DataArray, test_var: xr.DataArray) -> bool:
    """Compara shape, tipo y valores de una variable comun."""
    variable_ok = True

    ref_values = reference_var.values
    test_values = test_var.values

    print(f"  Variable {variable_name}:")

    if reference_var.shape != test_var.shape:
        print(f"    Shape: SHAPE DIFF reference={reference_var.shape}, test={test_var.shape}")
        return False

    print(f"    Shape: OK {reference_var.shape}")

    if reference_var.dtype == test_var.dtype:
        print(f"    Data type: OK {reference_var.dtype}")
    else:
        variable_ok = False
        print(f"    Data type: DIFFERENT reference={reference_var.dtype}, test={test_var.dtype}")

    if is_numeric_array(ref_values) and is_numeric_array(test_values):
        max_abs_diff = float(np.nanmax(np.abs(ref_values - test_values))) if ref_values.size else 0.0
        variable_rmse = rmse(ref_values, test_values)
        max_rel_diff = max_relative_difference(ref_values, test_values)

        values_ok = np.allclose(
            ref_values,
            test_values,
            atol=ABS_TOL,
            rtol=REL_TOL,
            equal_nan=True,
        )

        print(f"    Maximum absolute difference: {max_abs_diff:.12e}")
        print(f"    RMSE: {variable_rmse:.12e}")
        print(f"    Maximum relative difference: {max_rel_diff:.12e}")
        print(f"    Values: {'OK' if values_ok else 'DIFFERENT'}")

        variable_ok = variable_ok and values_ok
    else:
        values_ok = arrays_equal_non_numeric(ref_values, test_values)
        print(f"    Non-numeric values: {'OK' if values_ok else 'DIFFERENT'}")
        variable_ok = variable_ok and values_ok

    return variable_ok


def compare_file(reference_file: Path, test_file: Path) -> str:
    """Compara dos archivos NetCDF y devuelve OK o DIFFERENT."""
    print("=" * 80)
    print(f"FILE: {reference_file.name}")

    file_ok = True

    with xr.open_dataset(reference_file) as reference_ds, xr.open_dataset(test_file) as test_ds:
        print("Dimensions:")
        file_ok = compare_dimensions(reference_ds, test_ds) and file_ok

        reference_variables = set(reference_ds.variables)
        test_variables = set(test_ds.variables)

        only_reference = sorted(reference_variables - test_variables)
        only_test = sorted(test_variables - reference_variables)
        common_variables = sorted(reference_variables & test_variables)

        if only_reference:
            file_ok = False
            print(f"Variables only in reference: {only_reference}")

        if only_test:
            file_ok = False
            print(f"Variables only in test: {only_test}")

        print("Common variables:")
        for variable_name in common_variables:
            variable_ok = compare_variable(
                variable_name,
                reference_ds[variable_name],
                test_ds[variable_name],
            )
            file_ok = variable_ok and file_ok

    status = "OK" if file_ok else "DIFFERENT"
    print(f"FILE RESULT: {'PASSED' if file_ok else 'FAILED'}")
    return status


def run_crossvalidation() -> None:
    """Compare only the eight original L1B products in the configured directories."""
    print("L1B NETCDF CROSS-VALIDATION")
    print(f"Reference directory: {REFERENCE_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Tolerances: ABS_TOL={ABS_TOL}, REL_TOL={REL_TOL}")
    print("Scope: l1b_toa and l1b_toa_eq for four VNIR bands; no output_dia2_noecualiz or E2E data.")
    print()

    expected_names = [
        f"{product}_VNIR-{band}.nc"
        for product in ("l1b_toa", "l1b_toa_eq")
        for band in range(4)
    ]
    results = {}
    for file_name in expected_names:
        reference_file = REFERENCE_DIR / file_name
        test_file = TEST_DIR / file_name
        if not reference_file.is_file() or not test_file.is_file():
            print("=" * 80)
            print(f"FILE: {file_name}")
            if not reference_file.is_file():
                print("MISSING IN REFERENCE")
            if not test_file.is_file():
                print("MISSING IN TEST")
            print("FILE RESULT: MISSING")
            results[file_name] = "MISSING"
            continue
        results[file_name] = compare_file(reference_file, test_file)

    ok_count = sum(1 for status in results.values() if status == "OK")
    different_count = sum(1 for status in results.values() if status == "DIFFERENT")
    missing_count = sum(1 for status in results.values() if status == "MISSING")

    print("=" * 80)
    print("GLOBAL SUMMARY")
    print()
    print("Status of each expected NetCDF file:")
    display_status = {"OK": "PASSED", "DIFFERENT": "FAILED", "MISSING": "MISSING"}
    for file_name in sorted(results):
        print(f"  {file_name}: {display_status[results[file_name]]}")

    print()
    print(f"Total files compared: {len(results)}")
    print(f"PASSED: {ok_count}")
    print(f"FAILED: {different_count}")
    print(f"MISSING: {missing_count}")

    all_ok = (
        len(results) == 8
        and ok_count == len(results)
        and different_count == 0
        and missing_count == 0
    )
    print()
    if all_ok:
        print("GLOBAL RESULT: PASSED (all eight expected files passed)")
    else:
        print("GLOBAL RESULT: FAILED (differences or missing files)")


if __name__ == "__main__":
    run_crossvalidation()
