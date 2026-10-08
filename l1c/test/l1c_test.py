from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from netCDF4 import Dataset
from common.io.readGeodetic import readGeodetic

BASE_DIR = Path(r"C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri\EODP\EODP_TER_2021")

GM_DIR = BASE_DIR / "EODP-TS-L1C" / "input" / "gm_alt100_act_150"
L1C_DIR = BASE_DIR / "EODP-TS-L1C" / "output_test"

bands = ["VNIR-0", "VNIR-1", "VNIR-2", "VNIR-3"]

lat, lon = readGeodetic(str(GM_DIR), "geolocation.nc")

# L1B vs L1C grid for each band
for band in bands:

    with Dataset(L1C_DIR / f"l1c_toa_{band}.nc") as ds:
        lat_l1c = np.asarray(ds.variables["lat"][:])
        lon_l1c = np.asarray(ds.variables["lon"][:])

    plt.figure(figsize=(12, 7))
    plt.plot(lon.ravel(), lat.ravel(), '.r', markersize=2, label='L1B')
    plt.plot(lon_l1c, lat_l1c, '.b', markersize=3, label='L1C MGRS')
    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    plt.title('Projection on ground')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(L1C_DIR / f"projection_on_ground_{band}.png", dpi=300)

# SSD central row (ACT)
row = lat.shape[0] // 2
col = lat.shape[1] // 2

ssd_row = np.zeros(lat.shape[1] - 1)

for j in range(lat.shape[1] - 1):
    p1 = (float(lat[row, j]), float(lon[row, j]))
    p2 = (float(lat[row, j + 1]), float(lon[row, j + 1]))
    ssd_row[j] = haversine(p1, p2, unit=Unit.METERS)

plt.figure(figsize=(10, 5))
plt.plot(ssd_row)
plt.xlabel('Column (ACT)')
plt.ylabel('SSD [m]')
plt.title(f'Spatial Sampling Distance - Central Row {row}')
plt.grid(True)
plt.tight_layout()
plt.savefig(L1C_DIR / "ssd_central_row.png", dpi=300)

# SSD central column (ALT)
ssd_col = np.zeros(lat.shape[0] - 1)

for i in range(lat.shape[0] - 1):
    p1 = (float(lat[i, col]), float(lon[i, col]))
    p2 = (float(lat[i + 1, col]), float(lon[i + 1, col]))
    ssd_col[i] = haversine(p1, p2, unit=Unit.METERS)

plt.figure(figsize=(10, 5))
plt.plot(ssd_col)
plt.xlabel('Row (ALT)')
plt.ylabel('SSD [m]')
plt.title(f'Spatial Sampling Distance - Central Column {col}')
plt.grid(True)
plt.tight_layout()
plt.savefig(L1C_DIR / "ssd_central_column.png", dpi=300)

plt.show()