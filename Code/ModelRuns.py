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


def create_file(params, labs, method):

    path = "/Report/" + method + "_"
    if params['lagged_regs']:
        path += 'BB_'
    else:
        path += ['POLS_', 'FE_'][params['FE']]
        path += ['Incidence_', 'Onset_'][params['onset']]

    path += labs[params['dep_var']].replace(" ", "")

    if params['interactions'] is not None:
        path += "_BY" + labs[params['interactions'][0]]

    if params['lagged_regs']:
        path += "_Iter" + str(params['iterations'])
        path += "_LagDepth" + str(params['max_lags'])

    if method == 'Regress':
        path += "_" + str(int(params['fit_year']))
        path += ".tex"
    else:
        path += ".png"

    return(path)


if sections['Run_Regs']:

    # we use ordered dicts to store estimation results
    rep_armed = OrderedDict()
    rep_civil = OrderedDict()
    pool_dict = OrderedDict()
    fe_dict = OrderedDict()
    bb_armed_dict = OrderedDict()
    bb_civil_dict = OrderedDict()
    bb_deaths_dict = OrderedDict()

    print("============================================================")
    print("Replicating Basic Regression Models")
    print("============================================================")

    # Run the models for replication
    rep_armed['POLSOnset'] = run_model(
        master, params['pool']['armed']['onset']['Init'])
    rep_armed['FEOnset'] = run_model(
        master, params['fe']['armed']['onset']['Init'])
    rep_armed['POLSIncidence'] = run_model(
        master, params['pool']['armed']['incidence']['Init'])
    rep_armed['FEIncidence'] = run_model(
        master, params['fe']['armed']['incidence']['Init'])

    rep_file = currDir + '/Report/Regress_Replication_ArmedConflict_2013.tex'
    out_latex(rep_armed, all_labs, params['pool']
              ['armed']['onset']['Init'], rep_file)

    rep_civil['POLSOnset'] = run_model(
        master, params['pool']['civil']['onset']['Init'])
    rep_civil['FEOnset'] = run_model(
        master, params['fe']['civil']['onset']['Init'])
    rep_civil['POLSIncidence'] = run_model(
        master, params['pool']['civil']['incidence']['Init'])
    rep_civil['FEIncidence'] = run_model(
        master, params['fe']['civil']['incidence']['Init'])

    rep_file = currDir + '/Report/Regress_Replication_CivilWar_2013.tex'
    out_latex(rep_civil, all_labs, params['pool']
              ['civil']['onset']['Init'], rep_file)

    # 1. Run Pooled OLS Model with all interaction variants
    print("============================================================")
    print("Running Pooled Model Interaction Comparison")
    print("============================================================")
    pool_dict['Initial'] = run_model(
        master, params['pool']['armed']['onset']['Init'])
    pool_dict['Child'] = run_model(
        master, params['pool']['armed']['onset']['child'])
    pool_dict['Democ'] = run_model(
        master, params['pool']['armed']['onset']['democ'])
    pool_dict['GDP'] = run_model(
        master, params['pool']['armed']['onset']['gdp'])
    pool_dict['Good'] = run_model(
        master, params['pool']['armed']['onset']['good'])

    print("============================================================")
    print("Outputting Pooled Interaction Comparison to Latex")
    print("============================================================")
    pool_file = currDir + \
        create_file(params['pool']['armed']['onset']['Init'], labs, 'Regress')
    out_latex(pool_dict, all_labs, params['pool']
              ['armed']['onset']['Init'], pool_file)

    # 1. Run FE Model with all interaction variants
    print("============================================================")
    print("Running FE Model Interaction Comparison")
    print("============================================================")
    fe_dict['Initial'] = run_model(
        master, params['fe']['armed']['onset']['Init'])
    fe_dict['Child'] = run_model(
        master, params['fe']['armed']['onset']['child'])
    fe_dict['Democ'] = run_model(
        master, params['fe']['armed']['onset']['democ'])
    fe_dict['GDP'] = run_model(
        master, params['fe']['armed']['onset']['gdp'])
    fe_dict['Good'] = run_model(
        master, params['fe']['armed']['onset']['good'])

    print("============================================================")
    print("Outputting FE Interaction Comparison to Latex")
    print("============================================================")
    fe_file = currDir + \
        create_file(params['fe']['armed']['onset']['Init'], labs, 'Regress')
    out_latex(fe_dict, all_labs, params['fe']
              ['armed']['onset']['Init'], fe_file)

    # 1. Run Blundell-Bond Model with all interaction variants
    print("============================================================")
    print("Running Blundell-Bond Model")
    print("============================================================")
    bb_armed_dict['Initial'] = blundell_bond(
        master, params['bb']['armed']['Init'])
    bb_civil_dict['Initial'] = blundell_bond(
        master, params['bb']['civil']['Init'])
    bb_deaths_dict['Initial'] = blundell_bond(
        master, params['bb']['deaths']['Init'])

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Child Mortality")
    print("============================================================")
    bb_armed_dict['Child'] = blundell_bond(
        master, params['bb']['armed']['child'])
    bb_civil_dict['Child'] = blundell_bond(
        master, params['bb']['civil']['child'])
    bb_deaths_dict['Child'] = blundell_bond(
        master, params['bb']['deaths']['child'])

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Democracy")
    print("============================================================")
    bb_armed_dict['Democ'] = blundell_bond(
        master, params['bb']['armed']['democ'])
    bb_civil_dict['Democ'] = blundell_bond(
        master, params['bb']['civil']['democ'])
    bb_deaths_dict['Democ'] = blundell_bond(
        master, params['bb']['deaths']['democ'])

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with GDP")
    print("============================================================")
    bb_armed_dict['GDP'] = blundell_bond(
        master, params['bb']['armed']['gdp'])
    bb_civil_dict['GDP'] = blundell_bond(
        master, params['bb']['civil']['gdp'])
    bb_deaths_dict['GDP'] = blundell_bond(
        master, params['bb']['deaths']['gdp'])

    print("============================================================")
    print("Running Blundell-Bond Model Interacted with Good Index")
    print("============================================================")
    bb_armed_dict['Good'] = blundell_bond(
        master, params['bb']['armed']['good'])
    bb_civil_dict['Good'] = blundell_bond(
        master, params['bb']['civil']['good'])
    bb_deaths_dict['Good'] = blundell_bond(
        master, params['bb']['deaths']['good'])

    print("============================================================")
    print("Outputting Blundell-Bond Models to Latex")
    print("============================================================")
    bb_armed_file = currDir + \
        create_file(params['bb']['armed']['Init'], labs, "Regress")
    bb_civil_file = currDir + \
        create_file(params['bb']['civil']['Init'], labs, "Regress")
    bb_deaths_file = currDir + \
        create_file(params['bb']['deaths']['Init'], labs, "Regress")

    out_latex(bb_armed_dict, all_labs,
              params['bb']['armed']['Init'], bb_armed_file)
    out_latex(bb_civil_dict, all_labs,
              params['bb']['civil']['Init'], bb_civil_file)
    out_latex(bb_deaths_dict, all_labs,
              params['bb']['deaths']['Init'], bb_deaths_file)

if sections['Run_ROC']:

    print("============================================================")
    print("Replicating Basic ROC Curves")
    print("============================================================")

    file = currDir + create_file(params['fe']
                                 ['armed']['onset']['Init'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['onset']['Init'], file)

    file = currDir + \
        create_file(params['fe']['armed']['incidence']['Init'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['incidence']['Init'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['onset']['Init'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['onset']['Init'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['incidence']['Init'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['incidence']['Init'], file)

    print("============================================================")
    print("FE Model Interacted with Child Mortality ROC Curve")
    print("============================================================")

    file = currDir + create_file(params['fe']
                                 ['armed']['onset']['child'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['onset']['child'], file)

    file = currDir + \
        create_file(params['fe']['armed']['incidence']['child'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['incidence']['child'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['onset']['child'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['onset']['child'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['incidence']['child'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['incidence']['child'], file)

    print("============================================================")
    print("FE Model Interacted with Democracy ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['fe']
                                 ['armed']['onset']['democ'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['onset']['democ'], file)

    file = currDir + \
        create_file(params['fe']['armed']['incidence']['democ'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['incidence']['democ'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['onset']['democ'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['onset']['democ'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['incidence']['democ'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['incidence']['democ'], file)

    print("============================================================")
    print("FE Model Interacted with GDP ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['fe']
                                 ['armed']['onset']['gdp'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['onset']['gdp'], file)

    file = currDir + \
        create_file(params['fe']['armed']['incidence']['gdp'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['incidence']['gdp'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['onset']['gdp'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['onset']['gdp'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['incidence']['gdp'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['incidence']['gdp'], file)

    print("============================================================")
    print("FE Model Interacted with Good Index ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['fe']
                                 ['armed']['onset']['good'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['onset']['good'], file)

    file = currDir + \
        create_file(params['fe']['armed']['incidence']['good'], labs, 'ROC')
    compute_roc(master, params['fe']['armed']['incidence']['good'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['onset']['good'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['onset']['good'], file)

    file = currDir + create_file(params['fe']
                                 ['civil']['incidence']['good'], labs, 'ROC')
    compute_roc(master, params['fe']
                ['civil']['incidence']['good'], file)

    print("============================================================")
    print("Basic Blundell-Bond Model ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['bb']['armed']['Init'], labs, 'ROC')
    compute_roc(master, params['bb']['armed']['Init'], file)

    file = currDir + create_file(params['bb']['civil']['Init'], labs, 'ROC')
    compute_roc(master, params['bb']['civil']['Init'], file)

    print("============================================================")
    print("Blundell-Bond Model Interacted with Child Mortality ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['bb']['armed']['child'], labs, 'ROC')
    compute_roc(master, params['bb']['armed']['child'], file)

    file = currDir + create_file(params['bb']['civil']['child'], labs, 'ROC')
    compute_roc(master, params['bb']['civil']['child'], file)

    print("============================================================")
    print("Blundell-Bond Model Interacted with Democracy ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['bb']['armed']['democ'], labs, 'ROC')
    compute_roc(master, params['bb']['armed']['democ'], file)

    file = currDir + create_file(params['bb']['civil']['democ'], labs, 'ROC')
    compute_roc(master, params['bb']['civil']['democ'], file)

    print("============================================================")
    print("Blundell-Bond Model Interacted with GDP ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['bb']['armed']['gdp'], labs, 'ROC')
    compute_roc(master, params['bb']['armed']['gdp'], file)

    file = currDir + create_file(params['bb']['civil']['gdp'], labs, 'ROC')
    compute_roc(master, params['bb']['civil']['gdp'], file)

    print("============================================================")
    print("Blundell-Bond Model Interacted with Good Index ROC Curve")
    print("============================================================")
    file = currDir + create_file(params['bb']['armed']['good'], labs, 'ROC')
    compute_roc(master, params['bb']['armed']['good'], file)

    file = currDir + create_file(params['bb']['civil']['good'], labs, 'ROC')
    compute_roc(master, params['bb']['civil']['good'], file)
