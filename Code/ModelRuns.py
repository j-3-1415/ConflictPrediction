##########################################################################
# In this script different model runs can be executed and evaluated.
# Use this script to try out different model runs and compare them.
##########################################################################

# import data and functions
from Code.DataPrep import *
from Code.ModelFunc import *
from Code.ModelParams import *

# import libraries
from collections import OrderedDict

# we use ordered dicts to store estimation results
res_dict = OrderedDict()

# 1. Run: Pooled and FE model, bdbest25, no interactions
res_dict['Pooled'] = run_model(master, pooled_bdbest25)
res_dict['FE'] = run_model(master, fe_bdbest25)

# 2. Run: Pooled and FE model, bdbest25,  interactions
res_dict['PooledInteract'] = run_model(master, pooled_bdbest25_child)
res_dict['FEInteract'] = run_model(master, fe_bdbest25_child)

# export 1. Run and 2. Run
compare_file = currDir + "/Report/OLS_armed.tex"
# Andy used last set of model params for export?
out_latex(res_dict, all_labs, fe_bdbest25_child, compare_file, "custom")


# 3. Run: Pooled and FE model, bdbest1000, no interactions
res_dict['Pooled'] = run_model(master, pooled_bdbest1000)
res_dict['FE'] = run_model(master, fe_bdbest1000)

# 4. Run: Pooled and FE model, bdbest1000,  interactions
res_dict['PooledInteract'] = run_model(master, pooled_bdbest1000_child)
res_dict['FEInteract'] = run_model(master, fe_bdbest1000_child)

# export 3. Run and 4. Run
compare_file = currDir + "/Report/OLS_civil.tex"
# Andy used last set of model params for export?
out_latex(res_dict, all_labs, fe_bdbest1000_child, compare_file, "custom")


# 5. Run: Blundell bond bdbest25
gmm_dict = OrderedDict()
gmm_dict['GMM'] = blundell_bond(master, bb_params_bdbest25)

gmm_file = currDir + "/Report/GMM_armed.tex"
out_latex(gmm_dict, all_labs, bb_params_bdbest25, gmm_file, "custom")

# 5. Run: Blundell bond bdbest1000
gmm_dict = OrderedDict()
gmm_dict['GMM'] = blundell_bond(master, bb_params_bdbest1000)

gmm_file = currDir + "/Report/GMM_civil.tex"
out_latex(gmm_dict, all_labs, bb_params_bdbest1000, gmm_file, "custom")


# Export exemplary ROC for baseline fe bdbest 26
file = currDir + "/Report/ROC_FE.png"
compute_roc(master, fe_bdbest25, file)