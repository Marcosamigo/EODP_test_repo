from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


ABS_TOL = 1e-8
REL_TOL = 1e-6

REFERENCE_DIR = Path(
    r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\output"
)

TEST_DIR = Path(
    r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\output_test"
)

BANDS = range(4)


def expected_netcdf_names() -> list[str]:
    """NetCDF del ISM que aparecen como comparables en ambas carpetas."""
    prefixes = [
        "ism_toa_detection",
        "ism_toa_ds",
        "ism_toa_e",
        "ism_toa_isrf",
        "ism_toa_optical",
        "ism_toa_prnu",
        "ism_toa",
    ]

    return [
        f"{prefix}_VNIR-{band}.nc"
        for prefix in prefixes
        for band in BANDS
    ]


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
        print(f"  Dimensiones solo en referencia: {only_reference}")

    if only_test:
        dimensions_ok = False
        print(f"  Dimensiones solo en test: {only_test}")

    for dim in common_dims:
        ref_size = reference_dims[dim]
        test_size = test_dims[dim]
        if ref_size == test_size:
            print(f"  Dimension {dim}: OK ({ref_size})")
        else:
            dimensions_ok = False
            print(f"  Dimension {dim}: DIFFERENT referencia={ref_size}, test={test_size}")

    return dimensions_ok


def compare_variable(variable_name: str, reference_var: xr.DataArray, test_var: xr.DataArray) -> bool:
    """Compara shape, tipo y valores de una variable comun."""
    variable_ok = True

    ref_values = reference_var.values
    test_values = test_var.values

    print(f"  Variable {variable_name}:")

    if reference_var.shape != test_var.shape:
        print(f"    Shape: SHAPE DIFF referencia={reference_var.shape}, test={test_var.shape}")
        return False

    print(f"    Shape: OK {reference_var.shape}")

    if reference_var.dtype == test_var.dtype:
        print(f"    Tipo: OK {reference_var.dtype}")
    else:
        variable_ok = False
        print(f"    Tipo: DIFFERENT referencia={reference_var.dtype}, test={test_var.dtype}")

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

        print(f"    Diferencia maxima absoluta: {max_abs_diff:.12e}")
        print(f"    RMSE: {variable_rmse:.12e}")
        print(f"    Diferencia relativa maxima: {max_rel_diff:.12e}")
        print(f"    Valores: {'OK' if values_ok else 'DIFFERENT'}")

        variable_ok = variable_ok and values_ok
    else:
        values_ok = arrays_equal_non_numeric(ref_values, test_values)
        print(f"    Valores no numericos: {'OK' if values_ok else 'DIFFERENT'}")
        variable_ok = variable_ok and values_ok

    return variable_ok


def compare_netcdf_file(reference_file: Path, test_file: Path) -> str:
    """Compara dos archivos NetCDF y devuelve OK o DIFFERENT."""
    if xr is None:
        raise ImportError("Para comparar NetCDF necesitas instalar xarray: pip install xarray netcdf4")

    print("=" * 80)
    print(f"ARCHIVO NETCDF: {reference_file.name}")

    file_ok = True

    with xr.open_dataset(reference_file) as reference_ds, xr.open_dataset(test_file) as test_ds:
        print("Dimensiones:")
        file_ok = compare_dimensions(reference_ds, test_ds) and file_ok

        reference_variables = set(reference_ds.variables)
        test_variables = set(test_ds.variables)

        only_reference = sorted(reference_variables - test_variables)
        only_test = sorted(test_variables - reference_variables)
        common_variables = sorted(reference_variables & test_variables)

        if only_reference:
            file_ok = False
            print(f"Variables solo en referencia: {only_reference}")

        if only_test:
            file_ok = False
            print(f"Variables solo en test: {only_test}")

        print("Variables comunes:")
        for variable_name in common_variables:
            variable_ok = compare_variable(
                variable_name,
                reference_ds[variable_name],
                test_ds[variable_name],
            )
            file_ok = variable_ok and file_ok

    status = "OK" if file_ok else "DIFFERENT"
    print(f"RESULTADO DEL ARCHIVO: {'OK' if file_ok else 'HAY DIFERENCIAS'}")
    return status


def compare_expected_file(
    reference_name: str,
    test_name: str,
    compare_function,
) -> str:
    """Comprueba existencia y compara un par de archivos esperado."""
    reference_file = REFERENCE_DIR / reference_name
    test_file = TEST_DIR / test_name

    if not reference_file.exists():
        print("=" * 80)
        print(f"ARCHIVO: {reference_name}")
        print("MISSING EN REFERENCIA")
        print("RESULTADO DEL ARCHIVO: HAY DIFERENCIAS")
        return "MISSING_REFERENCE"

    if not test_file.exists():
        print("=" * 80)
        print(f"ARCHIVO: {reference_name}  <->  {test_name}")
        print("MISSING EN TEST")
        print("RESULTADO DEL ARCHIVO: HAY DIFERENCIAS")
        return "MISSING_TEST"

    return compare_function(reference_file, test_file)


def print_unmatched_files(compared_reference: set[str], compared_test: set[str]) -> None:
    """Lista archivos que existen en las carpetas pero no forman parte de los pares comparados."""
    reference_files = {
        path.name
        for path in REFERENCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".nc"
    }
    test_files = {
        path.name
        for path in TEST_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".nc"
    }

    only_reference = sorted(reference_files - compared_reference)
    only_test = sorted(test_files - compared_test)

    print()
    print("Archivos no comparados solo en referencia:")
    if only_reference:
        for file_name in only_reference:
            print(f"  {file_name}")
    else:
        print("  Ninguno")

    print("Archivos no comparados solo en test:")
    if only_test:
        for file_name in only_test:
            print(f"  {file_name}")
    else:
        print("  Ninguno")


def print_global_summary(results: dict[str, str], compared_reference: set[str], compared_test: set[str]) -> None:
    """Imprime resumen global de la cross-validation."""
    ok_count = sum(1 for status in results.values() if status == "OK")
    different_count = sum(1 for status in results.values() if status == "DIFFERENT")
    missing_count = sum(1 for status in results.values() if status.startswith("MISSING"))

    print("=" * 80)
    print("RESUMEN GLOBAL")
    print()
    print("Estado por archivo:")
    for file_name in sorted(results):
        print(f"  {file_name}: {results[file_name]}")

    print()
    print(f"Numero total de pares comparados: {len(results)}")
    print(f"Numero de archivos OK: {ok_count}")
    print(f"Numero de archivos diferentes: {different_count}")
    print(f"Numero de archivos ausentes: {missing_count}")

    print_unmatched_files(compared_reference, compared_test)

    all_ok = (
        len(results) > 0
        and ok_count == len(results)
        and different_count == 0
        and missing_count == 0
    )

    print()
    if all_ok:
        print("RESULTADO GLOBAL: CROSS-VALIDATION CORRECTA")
    else:
        print("RESULTADO GLOBAL: EXISTEN DIFERENCIAS")


def run_crossvalidation() -> None:
    """Ejecuta la cross-validation entre los outputs NetCDF del ISM."""
    print("CROSS-VALIDATION DE OUTPUTS NETCDF ISM")
    print(f"Carpeta de referencia: {REFERENCE_DIR}")
    print(f"Carpeta de test: {TEST_DIR}")
    print(f"Tolerancias: ABS_TOL={ABS_TOL}, REL_TOL={REL_TOL}")
    print()

    results = {}
    compared_reference = set()
    compared_test = set()

    for file_name in expected_netcdf_names():
        results[file_name] = compare_expected_file(
            file_name,
            file_name,
            compare_netcdf_file,
        )
        compared_reference.add(file_name)
        compared_test.add(file_name)

    print_global_summary(results, compared_reference, compared_test)


if __name__ == "__main__":
    run_crossvalidation()
