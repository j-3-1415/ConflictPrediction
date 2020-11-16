##########################################################################
# This script sets up parameters for different model runs
##########################################################################

from Code.DataPrep import interactions

print('Interactions terms available:')
print(interactions)
# got to data prep if you want to change the set of interaction terms that can be used

# baseline pooled bdbest25
pooled_bdbest25 = {
    "fit_year": 2013,  # Year for fitting the model
    "dep_var": "bdbest25",  # Civil War (1000) or Armed Conflict (25)
    "onset": True,  # Onset of Incidence of Conflict
    "all_indiv": True,  # Include all countries or not
    "FE": False,  # Pooled Model or Fixed Effects
    'interactions': None,  # Set of interaction vars (can be None)
    'dep_lags': 1,  # Number of lags in gmm
    'lagged_regs': False  # Whether blundell-bond is being used
}

# baseline fixed effects bdbest25
fe_bdbest25 = {
    "fit_year": 2013,  # Year for fitting the model
    "dep_var": "bdbest25",  # Civil War (1000) or Armed Conflict (25)
    "onset": True,  # Onset of Incidence of Conflict
    "all_indiv": True,  # Include all countries or not
    "FE": True,  # Pooled Model or Fixed Effects
    'interactions': None,  # Set of interaction vars (can be None)
    'dep_lags': 1,  # Number of lags in gmm
    'lagged_regs': False  # Whether blundell-bond is being used
}

# pooled with interaction bdbest25
pooled_bdbest25_child = pooled_bdbest25
pooled_bdbest25_child['interactions'] = interactions  # everything but interactions is the same

# fixed effects with interaction bdbest25
fe_bdbest25_child = fe_bdbest25
fe_bdbest25_child['interactions'] = interactions  # everything but interactions is the same

# baseline pooled bdbest1000
pooled_bdbest1000 = pooled_bdbest25
pooled_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same

# baseline fixed effects bdbest1000
fe_bdbest1000 = fe_bdbest25
fe_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same

# pooled with interaction bdbest1000
pooled_bdbest1000_child = pooled_bdbest25_child
pooled_bdbest1000_child['dep_var'] = 'bdbest1000'  # everything but dep var is the same

# fixed effects with interaction bdbest 1000
fe_bdbest1000_child = fe_bdbest25_child
fe_bdbest1000_child['dep_var'] = 'bdbest1000'  # everything but dep var is the same


# baseline blundell bond bdbest25
bb_params_bdbest25 = {
    "fit_year": 2013,  # Year for fitting the model
    "dep_var": "bdbest25",  # Civil War (1000) or Armed Conflict (25)
    "onset": True,  # Onset of Incidence of Conflict
    "all_indiv": True,  # Include all countries or not
    "FE": False,  # Pooled Model or Fixed Effects
    'interactions': None,  # Set of interaction vars (can be None)
    'max_lags': 3,  # Set max lags for instrumental variables
    'lagged_regs': True  # Define whether to use lagged labels
}

# baseline blundell bond bdbest1000
bb_params_bdbest1000 = bb_params_bdbest25
bb_params_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same

