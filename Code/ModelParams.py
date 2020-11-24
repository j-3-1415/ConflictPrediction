##########################################################################
# This script sets up parameters for different model runs
##########################################################################

from DataPrep import interactions

print('Interactions terms available to use:')
print(interactions)
# got to data prep if you want to change the set of interaction terms that can be used

# baseline pooled bdbest25
pool_armed = {
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
    'FD': False
}

pool_armed_child = pool_armed.copy()
pool_armed_goodex = pool_armed.copy()
pool_armed_democ = pool_armed.copy()
pool_armed_gdp = pool_armed.copy()

# baseline fixed effects bdbest25
fe_armed = {
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
    'FD': False
}

fe_armed_child = fe_armed.copy()
fe_armed_goodex = fe_armed.copy()
fe_armed_democ = fe_armed.copy()
fe_armed_gdp = fe_armed.copy()

# Start changing interactions terms to compare
pool_armed_child['interactions'] = ['childmortality']
pool_armed_goodex['interactions'] = ['avegoodex']
pool_armed_democ['interactions'] = ['democracy']
pool_armed_gdp['interactions'] = ['rgdpl']

# Start changing interactions terms to compare
fe_armed_child['interactions'] = ['childmortality']
fe_armed_goodex['interactions'] = ['avegoodex']
fe_armed_democ['interactions'] = ['democracy']
fe_armed_gdp['interactions'] = ['rgdpl']


# baseline blundell bond bdbest25
bb_armed = {
    "fit_year": 2013,  # Year for fitting the model
    "dep_var": "bdbest25",  # Civil War (1000) or Armed Conflict (25)
    "onset": True,  # Onset of Incidence of Conflict
    "all_indiv": True,  # Include all countries or not
    "FE": False,  # Pooled Model or Fixed Effects
    'interactions': None,  # Set of interaction vars (can be None)
    'max_lags': 3,  # Set max lags for instrumental variables
    'lagged_regs': True,  # Define whether to use lagged labels
    'iterations': 100,  # How many iterations for system gmm
    'topic_cols': ['theta' + str(i) for i in range(1, 15)],  # theta cols
    'weight_type': 'unadjusted',  # Type of gmm weighting matrix
    'FD': False
}

bb_armed_child = bb_armed.copy()
bb_armed_goodex = bb_armed.copy()
bb_armed_democ = bb_armed.copy()
bb_armed_gdp = bb_armed.copy()

# blundell bond with interaction bdbest 1000
bb_armed_child = bb_armed.copy()
bb_armed_child['interactions'] = ['childmortality']
bb_armed_goodex['interactions'] = ['avegoodex']
bb_armed_democ['interactions'] = ['democracy']
bb_armed_gdp['interactions'] = ['rgdpl']
