##########################################################################
# In this script different model runs can be executed and evaluated.
# Use this script to try out different model runs and compare them.
##########################################################################

import warnings
from ModelParams import *
from ModelFunc import *
from DataPrep import *
from collections import OrderedDict
from tkinter import *

# import data and functions
# import libraries
warnings.filterwarnings('ignore')

global sections
sections = {}


class mainWindow(object):
    def __init__(self, master):
        self.master = master
        self.Selection = {'Run_Regs': BooleanVar(),
                          'Run_ROC': BooleanVar()}
        self.b = Checkbutton(master, text='Run Regressions',
                             variable=self.Selection['Run_Regs'])
        self.b.pack(anchor='w')
        self.b2 = Checkbutton(master, text='Run ROC Curves',
                              variable=self.Selection['Run_ROC'])
        self.b2.pack(anchor='w')
        self.b3 = Button(master, text='Finish Selection',
                         command=lambda: self.finish()).pack()

    def finish(self):
        sections['Run_Regs'] = self.Selection['Run_Regs'].get()
        sections['Run_ROC'] = self.Selection['Run_ROC'].get()
        self.master.destroy()


root = Tk()
root.geometry("350x75")
root.title("Select Sections to Run")
m = mainWindow(root)
root.mainloop()

if sections['Run_Regs']:

        # we use ordered dicts to store estimation results
    pool_dict = OrderedDict()
    fe_dict = OrderedDict()
    bb_dict = OrderedDict()

    # 1. Run Pooled OLS Model with all interaction variants
    print("============================================================")
    print("Running Pooled Model")
    print("============================================================")
    pool_dict['Initial'] = run_model(master, pool_armed)

    print("============================================================")
    print("Running Pooled Model Interacted with Child Mortality")
    print("============================================================")
    pool_dict['ChildMortality'] = run_model(master, pool_armed_child)

    print("============================================================")
    print("Running Pooled Model Interacted with Democracy")
    print("============================================================")
    pool_dict['Democracy'] = run_model(master, pool_armed_democ)

    print("============================================================")
    print("Running Pooled Model Interacted with GDP")
    print("============================================================")
    pool_dict['GDP'] = run_model(master, pool_armed_gdp)

    print("============================================================")
    print("Running Pooled Model Interacted with Good Index")
    print("============================================================")
    pool_dict['GoodIndex'] = run_model(master, pool_armed_goodex)

    print("============================================================")
    print("Outputting Pooled OLS Models to Latex")
    print("============================================================")
    pool_file = currDir + "/Report/Pooled_armed.tex"
    out_latex(pool_dict, all_labs, pool_armed, pool_file, 'custom')

    # 1. Run FE Model with all interaction variants
    print("============================================================")
    print("Running FE Model")
    print("============================================================")
    fe_dict['Initial'] = run_model(master, fe_armed)

    print("============================================================")
    print("Running FE Model Interacted with Child Mortality")
    print("============================================================")
    fe_dict['ChildMortality'] = run_model(master, fe_armed_child)

    print("============================================================")
    print("Running FE Model Interacted with Democracy")
    print("============================================================")
    fe_dict['Democracy'] = run_model(master, fe_armed_democ)

    print("============================================================")
    print("Running FE Model Interacted with GDP")
    print("============================================================")
    fe_dict['GDP'] = run_model(master, fe_armed_gdp)

    print("============================================================")
    print("Running FE Model Interacted with Good Index")
    print("============================================================")
    fe_dict['GoodIndex'] = run_model(master, fe_armed_goodex)

    print("============================================================")
    print("Outputting FE Models to Latex")
    print("============================================================")
    fe_file = currDir + "/Report/FE_armed.tex"
    out_latex(fe_dict, all_labs, fe_armed, fe_file, 'custom')

    # 1. Run Blundell-Bond Model with all interaction variants
    print("============================================================")
    print("Running Blundell-Bond Model")
    print("============================================================")
    bb_dict['Initial'] = blundell_bond(master, bb_armed)

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Child Mortality")
    print("============================================================")
    bb_dict['ChildMortality'] = blundell_bond(master, bb_armed_child)

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Democracy")
    print("============================================================")
    bb_dict['Democracy'] = blundell_bond(master, bb_armed_democ)

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with GDP")
    print("============================================================")
    bb_dict['GDP'] = blundell_bond(master, bb_armed_gdp)

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Good Index")
    print("============================================================")
    bb_dict['GoodIndex'] = blundell_bond(master, bb_armed_goodex)

    print("============================================================")
    print("Outputting Blundell-Bond Models to Latex")
    print("============================================================")
    bb_file = currDir + "/Report/BB_armed.tex"
    out_latex(bb_dict, all_labs, bb_armed, bb_file, 'custom')

if sections['Run_ROC']:

    # Export exemplary ROC for baseline fe bdbest 25
    print("============================================================")
    print("Computing Basic FE Model ROC Curve")
    print("============================================================")
    file = currDir + "/Report/ROC_FE.png"
    compute_roc(master, fe_armed, file)

    print("============================================================")
    print("Computing FE Model Interacted with Child Mortality ROC Curve")
    print("============================================================")
    file = currDir + "/Report/ROC_FE_child.png"
    compute_roc(master, fe_armed_child, file)

    print("============================================================")
    print("Computing FE Model Interacted with Democracy ROC Curve")
    print("============================================================")
    file = currDir + "/Report/ROC_FE_democ.png"
    compute_roc(master, fe_armed_democ, file)

    print("============================================================")
    print("Computing FE Model Interacted with GDP ROC Curve")
    print("============================================================")
    file = currDir + "/Report/ROC_FE_gdp.png"
    compute_roc(master, fe_armed_gdp, file)

    print("============================================================")
    print("Computing FE Model Interacted with Good Index ROC Curve")
    print("============================================================")
    file = currDir + "/Report/ROC_FE_goodex.png"
    compute_roc(master, fe_armed_goodex, file)

    print("============================================================")
    print("Computing Basic Blundell-Bond Model ROC Curve")
    print("============================================================")
    file = currDir + '/Report/ROC_BB.png'
    compute_roc(master, bb_armed, file)

    print("============================================================")
    print("Computing Blundell-Bond Model Interacted with Child Mortality ROC Curve")
    print("============================================================")
    file = currDir + '/Report/ROC_BB_child.png'
    compute_roc(master, bb_armed_child, file)

    print("============================================================")
    print("Computing Blundell-Bond Model Interacted with Democracy ROC Curve")
    print("============================================================")
    file = currDir + '/Report/ROC_BB_democ.png'
    compute_roc(master, bb_armed_democ, file)

    print("============================================================")
    print("Computing Blundell-Bond Model Interacted with GDP ROC Curve")
    print("============================================================")
    file = currDir + '/Report/ROC_BB_gdp.png'
    compute_roc(master, bb_armed_gdp, file)

    print("============================================================")
    print("Computing Blundell-Bond Model Interacted with Good Index ROC Curve")
    print("============================================================")
    file = currDir + '/Report/ROC_BB_goodex.png'
    compute_roc(master, bb_armed_goodex, file)
