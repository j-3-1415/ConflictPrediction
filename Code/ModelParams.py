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
    'lagged_regs': False,  # Whether blundell-bond is being used
    'iterations': 2,  # How many iterations for system gmm
    'topic_cols': ['theta' + str(i) for i in range(1, 15)],  # theta cols
    'weight_type': 'unadjusted',  # Type of gmm weighting matrix
    'FD': True
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
    'lagged_regs': False,  # Whether blundell-bond is being used
    'iterations': 2,  # How many iterations for system gmm
    'topic_cols': ['theta' + str(i) for i in range(1, 15)],  # theta cols
    'weight_type': 'unadjusted',  # Type of gmm weighting matrix
    'FD': True
}

# pooled with interaction bdbest25
pooled_bdbest25_child = pooled_bdbest25.copy()
# everything but interactions is the same
pooled_bdbest25_child['interactions'] = interactions

# fixed effects with interaction bdbest25
fe_bdbest25_child = fe_bdbest25.copy()
# everything but interactions is the same
fe_bdbest25_child['interactions'] = interactions

# baseline pooled bdbest1000
pooled_bdbest1000 = pooled_bdbest25.copy()
pooled_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same

# baseline fixed effects bdbest1000
fe_bdbest1000 = fe_bdbest25.copy()
fe_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same

# pooled with interaction bdbest1000
pooled_bdbest1000_child = pooled_bdbest25_child.copy()
# everything but dep var is the same
pooled_bdbest1000_child['dep_var'] = 'bdbest1000'

# fixed effects with interaction bdbest 1000
fe_bdbest1000_child = fe_bdbest25_child.copy()
# everything but dep var is the same
fe_bdbest1000_child['dep_var'] = 'bdbest1000'


# baseline blundell bond bdbest25
bb_bdbest25 = {
    "fit_year": 2013,  # Year for fitting the model
    "dep_var": "bdbest25",  # Civil War (1000) or Armed Conflict (25)
    "onset": True,  # Onset of Incidence of Conflict
    "all_indiv": True,  # Include all countries or not
    "FE": False,  # Pooled Model or Fixed Effects
    'interactions': None,  # Set of interaction vars (can be None)
    'max_lags': 2,  # Set max lags for instrumental variables
    'lagged_regs': True,  # Define whether to use lagged labels
    'iterations': 10,  # How many iterations for system gmm
    'topic_cols': ['theta' + str(i) for i in range(1, 15)],  # theta cols
    'weight_type': 'unadjusted',  # Type of gmm weighting matrix
    'FD': False
}

# blundell bond with interaction bdbest 1000
bb_bdbest25_child = bb_bdbest25.copy()
bb_bdbest25_child['interactions'] = [interactions[0]]

bb_bdbest25_FD = bb_bdbest25.copy()
bb_bdbest25_FD['FD'] = True

# baseline blundell bond bdbest1000
bb_bdbest1000 = bb_bdbest25.copy()
bb_bdbest1000['dep_var'] = 'bdbest1000'  # everything but dep var is the same
