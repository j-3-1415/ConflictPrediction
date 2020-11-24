##########################################################################
# In this script different model runs can be executed and evaluated.
# Use this script to try out different model runs and compare them.
##########################################################################

import warnings
from ModelParams import *
from ModelFunc import *
from DataPrep import *
from collections import OrderedDict

# import data and functions
# import libraries
warnings.filterwarnings('ignore')

# we use ordered dicts to store estimation results
pool_dict = OrderedDict()
fe_dict = OrderedDict()
bb_dict = OrderedDict()

# 1. Run Pooled OLS Model with all interaction variants
print("============================================================")
print("Running Pooled Model")
print("============================================================")
pool_dict['Pooled'] = run_model(master, pool_armed)

print("============================================================")
print("Running Pooled Model Interacted with Child Mortality")
print("============================================================")
pool_dict['Pooled_ChildMortality'] = run_model(master, pool_armed_child)

print("============================================================")
print("Running Pooled Model Interacted with Democracy")
print("============================================================")
pool_dict['Pooled_Democracy'] = run_model(master, pool_armed_democ)

print("============================================================")
print("Running Pooled Model Interacted with GDP")
print("============================================================")
pool_dict['Pooled_GDP'] = run_model(master, pool_armed_gdp)

print("============================================================")
print("Running Pooled Model Interacted with Good Index")
print("============================================================")
pool_dict['Pooled_GoodIndex'] = run_model(master, pool_armed_goodex)

print("============================================================")
print("Outputting Pooled OLS Models to Latex")
print("============================================================")
pool_file = currDir + "/Report/Pooled_armed.tex"
out_latex(pool_dict, all_labs, pool_armed, pool_file, 'custom')


# 1. Run FE Model with all interaction variants
print("============================================================")
print("Running FE Model")
print("============================================================")
fe_dict['FE'] = run_model(master, fe_armed)

print("============================================================")
print("Running FE Model Interacted with Child Mortality")
print("============================================================")
fe_dict['FE_ChildMortality'] = run_model(master, fe_armed_child)

print("============================================================")
print("Running FE Model Interacted with Democracy")
print("============================================================")
fe_dict['FE_Democracy'] = run_model(master, fe_armed_democ)

print("============================================================")
print("Running FE Model Interacted with GDP")
print("============================================================")
fe_dict['FE_GDP'] = run_model(master, fe_armed_gdp)

print("============================================================")
print("Running FE Model Interacted with Good Index")
print("============================================================")
fe_dict['FE_GoodIndex'] = run_model(master, fe_armed_goodex)

print("============================================================")
print("Outputting FE Models to Latex")
print("============================================================")
fe_file = currDir + "/Report/FE_armed.tex"
out_latex(fe_dict, all_labs, fe_armed, fe_file, 'custom')


# 1. Run Blundell-Bond Model with all interaction variants
print("============================================================")
print("Running Blundell-Bond Model")
print("============================================================")
bb_dict['BB'] = run_model(master, bb_armed)

print("============================================================")
print("Running Blundell-Bond Model Interacted with Child Mortality")
print("============================================================")
bb_dict['BB_ChildMortality'] = run_model(master, bb_armed_child)

print("============================================================")
print("Running Blundell-Bond Model Interacted with Democracy")
print("============================================================")
bb_dict['BB_Democracy'] = run_model(master, bb_armed_democ)

print("============================================================")
print("Running Blundell-Bond Model Interacted with GDP")
print("============================================================")
bb_dict['BB_GDP'] = run_model(master, bb_armed_gdp)

print("============================================================")
print("Running Blundell-Bond Model Interacted with Good Index")
print("============================================================")
bb_dict['BB_GoodIndex'] = run_model(master, bb_armed_goodex)

print("============================================================")
print("Outputting Blundell-Bond Models to Latex")
print("============================================================")
bb_file = currDir + "/Report/BB_armed.tex"
out_latex(bb_dict, all_labs, bb_armed, bb_file, 'custom')


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
