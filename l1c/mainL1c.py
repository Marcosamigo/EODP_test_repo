
# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_test_repo\\auxiliary'
# GM dir + L1B dir
indir = r'C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri\EODP\EODP-TS-E2E-20261022T152838Z-1-001\EODP-TS-E2E\gm_out,C:\Users\marco\OneDrive\Escritorio\UC3M\Master\3er Cuatri\EODP\EODP-TS-E2E-20261022T152838Z-1-001\EODP-TS-E2E\my_output_l1b'
outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP-TS-E2E-20261022T152838Z-1-001\\EODP-TS-E2E\\my_output_l1c"


# Initialise the ISM
myL1c = l1c(auxdir, indir, outdir)
myL1c.processModule()
