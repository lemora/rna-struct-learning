from scipy import constants

# global constants

VERBOSE: bool = False
DISPLAY: bool = False

# GAS_CONST = constants.R  # options: 1,k,R
K_B = constants.k * constants.N_A / 4184  # Boltzmann constant in kcal/mol/K
BASE_TEMP = 310.0  # 37 deg C
# BASE_TEMP = 270.0  # -3.15 deg C
TEMP = BASE_TEMP
