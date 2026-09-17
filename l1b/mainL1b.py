
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_test_repo\\auxiliary'
indir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-L1B\\input" # small scene
outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_TER_2021\\EODP-TS-L1B\\output_test_mio"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
