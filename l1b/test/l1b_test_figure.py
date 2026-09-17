from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ================================================================
# RUTAS
# ================================================================

BASE_DIR = Path(
    r"C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri"
    r"\EODP\EODP_TER_2021\EODP-TS-L1B"
)

# Datos equalizados
FILE_EQ = (
    BASE_DIR
    / "output_test_mio"
    / "l1b_toa_VNIR-0.nc"
)

# Datos NO equalizados
FILE_NO_EQ = (
    BASE_DIR
    / "output_dia2_noecualiz"
    / "l1b_toa_VNIR-0.nc"
)

# Ground truth
FILE_TRUTH = (
    BASE_DIR
    / "input"
    / "ism_toa_isrf_VNIR-0.nc"
)


# ================================================================
# CONFIGURACIÓN
# ================================================================

VARIABLE = "toa"

# La variable tiene shape (100, 150).
# El segundo eje corresponde a los 150 ACT pixels.
#
# Se selecciona una línea de la primera dimensión.
# Puedes cambiar este valor entre 0 y 99.
ALT_INDEX = 50


# ================================================================
# FUNCIÓN PARA LEER TOA
# ================================================================

def read_toa_profile(file_path, variable="toa", alt_index=50):
    """
    Lee la variable TOA de un NetCDF y extrae un perfil
    a lo largo de los ACT pixels.

    Para un array (100, 150):
        - primera dimensión: línea seleccionada
        - segunda dimensión: ACT pixels

    Devuelve un vector de 150 muestras.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo:\n{file_path}"
        )

    with xr.open_dataset(file_path) as ds:

        if variable not in ds.variables:
            raise ValueError(
                f"La variable '{variable}' no existe en:\n"
                f"{file_path}\n\n"
                f"Variables disponibles: {list(ds.variables)}"
            )

        data = np.asarray(
            ds[variable].values,
            dtype=np.float64
        )

        print()
        print(f"Archivo: {file_path.name}")
        print(f"Variable: {variable}")
        print(f"Dimensiones: {ds[variable].dims}")
        print(f"Shape: {data.shape}")

        # --------------------------------------------------------
        # Caso 1: ya es un vector
        # --------------------------------------------------------

        if data.ndim == 1:
            return data

        # --------------------------------------------------------
        # Caso 2: matriz 2D
        # --------------------------------------------------------

        if data.ndim == 2:

            if alt_index >= data.shape[0]:
                raise IndexError(
                    f"ALT_INDEX={alt_index} está fuera de rango.\n"
                    f"La primera dimensión tiene tamaño "
                    f"{data.shape[0]}."
                )

            profile = data[alt_index, :]

            print(
                f"Perfil seleccionado: fila {alt_index}"
            )
            print(
                f"Número de ACT pixels: {profile.size}"
            )

            return profile

        raise ValueError(
            f"No se esperaba una variable TOA con "
            f"{data.ndim} dimensiones."
        )


# ================================================================
# LEER LOS TRES PRODUCTOS
# ================================================================

toa_eq = read_toa_profile(
    FILE_EQ,
    VARIABLE,
    ALT_INDEX
)

toa_no_eq = read_toa_profile(
    FILE_NO_EQ,
    VARIABLE,
    ALT_INDEX
)

toa_truth = read_toa_profile(
    FILE_TRUTH,
    VARIABLE,
    ALT_INDEX
)


# ================================================================
# COMPROBAR TAMAÑOS
# ================================================================

print()
print("=" * 60)
print("COMPROBACIÓN DE TAMAÑOS")
print("=" * 60)

print(f"Equalizado     : {toa_eq.shape}")
print(f"No equalizado  : {toa_no_eq.shape}")
print(f"TRUTH           : {toa_truth.shape}")

if not (
    len(toa_eq)
    == len(toa_no_eq)
    == len(toa_truth)
):
    raise ValueError(
        "Los tres perfiles no tienen el mismo número de muestras."
    )


# ================================================================
# MÉTRICAS RESPECTO AL TRUTH
# ================================================================

valid_eq = (
    np.isfinite(toa_eq)
    & np.isfinite(toa_truth)
)

valid_no_eq = (
    np.isfinite(toa_no_eq)
    & np.isfinite(toa_truth)
)


error_eq = (
    toa_eq[valid_eq]
    - toa_truth[valid_eq]
)

error_no_eq = (
    toa_no_eq[valid_no_eq]
    - toa_truth[valid_no_eq]
)


rmse_eq = np.sqrt(
    np.mean(error_eq ** 2)
)

rmse_no_eq = np.sqrt(
    np.mean(error_no_eq ** 2)
)

mae_eq = np.mean(
    np.abs(error_eq)
)

mae_no_eq = np.mean(
    np.abs(error_no_eq)
)


print()
print("=" * 60)
print("ERRORES RESPECTO AL TRUTH")
print("=" * 60)

print(f"RMSE equalizado     : {rmse_eq:.6f}")
print(f"RMSE no equalizado  : {rmse_no_eq:.6f}")

print(f"MAE equalizado      : {mae_eq:.6f}")
print(f"MAE no equalizado   : {mae_no_eq:.6f}")


# ================================================================
# EJE ACT
# ================================================================

act_pixel = np.arange(
    len(toa_truth)
)


# ================================================================
# FIGURA 8.3
# ================================================================

plt.figure(
    figsize=(13, 6.5)
)


# Equalizado: negro
plt.plot(
    act_pixel,
    toa_eq,
    color="black",
    linewidth=1.5,
    label="TOA L1B with eq"
)


# No equalizado: rojo
plt.plot(
    act_pixel,
    toa_no_eq,
    color="red",
    linewidth=1.5,
    label="TOA L1B no eq"
)


# Truth: azul
plt.plot(
    act_pixel,
    toa_truth,
    color="blue",
    linewidth=2,
    label="TOA after the ISRF"
)


plt.title(
    "Effect of the Equalization for VNIR-0",
    fontsize=15
)

plt.xlabel(
    "ACT pixel [-]",
    fontsize=12
)

plt.ylabel(
    "TOA [mW/m²/sr]",
    fontsize=12
)

plt.grid(
    True,
    alpha=0.4
)

plt.legend(
    loc="upper left"
)

plt.tight_layout()


# ================================================================
# GUARDAR FIGURA
# ================================================================

OUTPUT_FIGURE = (
    BASE_DIR
    / "Figure_8_3_Equalization_VNIR-0.png"
)

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

print()
print(
    f"Figura guardada en:\n{OUTPUT_FIGURE}"
)


plt.show()