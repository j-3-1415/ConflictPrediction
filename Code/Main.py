################################################################################
# Main Script to Initiate the Scripts Correctly
################################################################################

import os  # Import for directory definitions
import sys  # Import to make sure scripts can be imported correctly
import pkgutil as pkg  # Get this library for listing current packages
from tkinter import *

global currDir

################################################################################
# Use the sys.executable to get the correct python version for executable
################################################################################

python = sys.executable

################################################################################
# Forcing the current directory
################################################################################

dirsplit = os.getcwd().split("/")
if 'mastercode' in dirsplit:
    currDir = os.path.abspath("/".\
        join(dirsplit[:dirsplit.index('mastercode') + 1]))
elif dirsplit[-1] == 'PanelData2020':
    currDir = os.path.abspath("/".join(dirsplit + ['mastercode']))
else:
    print("Please change to project directory before running code")
    sys.exit()
    
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
# Ask User Whether to Run Models, Plots or Both
################################################################################

global sections
sections = {}


class mainWindow(object):
    def __init__(self, master):
        self.master = master
        self.Selection = {'Run_Regs': BooleanVar(),
                          'Run_Plots': BooleanVar()}
        self.b = Checkbutton(master, text='Run Regressions/ROC Curves',
                             variable=self.Selection['Run_Regs'])
        self.b.pack(anchor='w')
        self.b2 = Checkbutton(master, text='Run Descriptive Plots',
                              variable=self.Selection['Run_Plots'])
        self.b2.pack(anchor='w')
        self.b3 = Button(master, text='Finish Selection',
                         command=lambda: self.finish()).pack()

    def finish(self):
        sections['Run_Regs'] = self.Selection['Run_Regs'].get()
        sections['Run_Plots'] = self.Selection['Run_Plots'].get()
        self.master.destroy()


root = Tk()
root.geometry("350x75")
root.title("Select Sections to Run")
m = mainWindow(root)
root.mainloop()

################################################################################
# Use os.system to run the Data Description code in command prompt
################################################################################

if sections['Run_Plots']:
    from DataDescr import *

################################################################################
# Use os.system to run the ModelRuns file in the command prompt
################################################################################

if sections['Run_Regs']:
    from ModelRuns import *
