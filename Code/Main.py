################################################################################
# Main Script to Initiate the Scripts Correctly
################################################################################

import os  # Import for directory definitions
import sys  # Import to make sure scripts can be imported correctly
import pkgutil as pkg #Get this library for listing current packages

################################################################################
# Use the sys.executable to get the correct python version for executable
################################################################################

python = sys.executable

################################################################################
# Forcing the current directory
################################################################################

dirsplit = os.getcwd().split("/")
currDir = os.path.abspath("/".join(dirsplit[:dirsplit.index('mastercode') + 1]))
os.chdir(currDir)

################################################################################
# Make sure that the scripts will look in the code file for script imports
################################################################################

sys.path.append('Code')

################################################################################
# Get the missing libraries needed to run this project
################################################################################

required = {'sklearn', 'statsmodels', 'collections', 'itertools', 'warnings',
            'linearmodels', 'matplotlib', 'numpy', 'pandas', 'tqdm',
            'plotly', 'seaborn', 'tkinter'}
installed = {i[1] for i in pkg.iter_modules()}

base = {i for i in sys.builtin_module_names}

missing = required - installed - base

################################################################################
# If the set of missing libraries is not empty, then install missing with pip
################################################################################

if missing:
	os.system(" ".join([python, '-m', 'pip', 'install'] + [i for i in missing]))

################################################################################
# Use os.system to run the Data Description code in command prompt
################################################################################

# os.system(python + ' Code/DataDescr.py')

################################################################################
# Use os.system to run the ModelRuns file in the command prompt
################################################################################

os.system(python + ' Code/ModelRuns.py')

