
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP_test_repo\\auxiliary'
indir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\EODP\\EODP-TS-E2E-20261022T152838Z-1-001\\EODP-TS-E2E\\sgm_out" # small scene
outdir = r"C:\\Users\\marco\\OneDrive\\Escritorio\\UC3M\\Master\\3er Cuatri\\EODP\\EODP-TS-E2E-20261022T152838Z-1-001\\EODP-TS-E2E\\my_output"


# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
