##########################################################################
# This script sets up parameters for different model runs
##########################################################################

from DataPrep import interactions

print('Interactions terms available to use:')
print(interactions)
# got to data prep if you want to change the set of interaction terms that can be used

# baseline pooled bdbest25
pool_armed_onset = {
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
}

# baseline fixed effects bdbest25
fe_armed_onset = {
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
}

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
    'iterations': 15,  # How many iterations for system gmm
    'topic_cols': ['theta' + str(i) for i in range(1, 15)],  # theta cols
    'weight_type': 'unadjusted',  # Type of gmm weighting matrix
    'dep_lags' : 1,
    'maxldep' : 3,
    'scaled' : ['rgdpl'],
    'cov_type' : 'robust'
}

dep_map = {'armed': 'bdbest25', 'civil': 'bdbest1000', 'deaths': 'bdbest'}
inter_map = {'child': 'childmortality', 'democ': 'democracy', 'gdp': 'rgdpl',
             'good': 'avegoodex'}
params = {}
for mod in ['pool', 'fe', 'bb']:
    params[mod] = {}
    for dep in ['armed', 'civil', 'deaths']:
        params[mod][dep] = {}
        if mod == 'bb':
            params[mod][dep]['Init'] = bb_armed.copy()
            params[mod][dep]['Init']['dep_var'] = dep_map[dep]

        for var in ['onset', 'incidence']:
            if mod == 'pool':
                params[mod][dep][var] = {}
                params[mod][dep][var]['Init'] = pool_armed_onset.copy()
                params[mod][dep][var]['Init']['dep_var'] = dep_map[dep]
                params[mod][dep][var]['Init']['onset'] = [
                    False, True][var == 'onset']
            elif mod == 'fe':
                params[mod][dep][var] = {}
                params[mod][dep][var]['Init'] = fe_armed_onset.copy()
                params[mod][dep][var]['Init']['dep_var'] = dep_map[dep]
                params[mod][dep][var]['Init']['onset'] = [
                    False, True][var == 'onset']
            for inter in ['child', 'democ', 'gdp', 'good']:
                if mod == 'bb':
                    params[mod][dep][inter] = params[mod][dep]['Init'].copy(
                    )
                    params[mod][dep][inter]['interactions'] = [inter_map[inter]]
                else:
                    params[mod][dep][var][inter] = params[mod][dep][var]['Init'].copy(
                    )
                    params[mod][dep][var][inter]['interactions'] = [
                        inter_map[inter]]
