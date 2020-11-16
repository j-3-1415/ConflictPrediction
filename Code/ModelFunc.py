##########################################################################
# This script introduces all functions necessary to
# 1. Run the models of interest
#   a. Pooled model, Fixed Effects
#   b. Blundell Bond
# 2. Use models for predictions
#   a. Compute Predictions
#   b. Return ROC curve
# 3. Output estimation results to latex
##########################################################################

# import data
from Code.DataPrep import *

# import libraries
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels import IVSystemGMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1a. Fixed Effects model

def run_model(data, params):

    ########################################################################
    # Input Notes
    ########################################################################
    # data should be master file
    # fit year is year t, before prediction year

    fit_year = params['fit_year']

    dep_var = params['dep_var']  # dep var should be bdbest25 or bdbest1000

    # onset is when there is no conflict in t but there is in t + 1
    # incidence is when there is conflict in t and conflict in t + 1
    onset = params['onset']

    # If not all_indiv, then removes counties with no conflict in the sample
    all_indiv = params['all_indiv']

    FE = params['FE']  # Boolean for Pooled OLS or Fixed Effects Panel

    # Get the list of variables to interact with the theta shares
    # This input diverges from the authors technique, so results will differ
    interactions = params['interactions']

    # The order of null filling and removal is done according to the authors
    #       code, changing the order changes the result
    ############################################################################
    # Input Notes
    ############################################################################

    # Get list of countries that are excluded for each run
    # Make a regression summary text file with each run

    data = data.copy(deep=True)  # Make a copy so dataframe not overwritten
    data = data[data['theta_year'] == (fit_year + 1)]

    # Define the column names of thetas to be used as regressors
    thetas = ["theta" + str(i) for i in range(1, 15)]

    # One before here means you are in t (one before the next period)
    data['one_before'] = data.groupby('countryid')[dep_var].shift(-1)

    if onset:  # Condition for whether this is onset or incidence, explained above
        data['one_before'] = np.where(
            (data['one_before'] == 1) & (data[dep_var] == 0), 1, 0)
    else:
        data['one_before'] = np.where(
            (data['one_before'] == 1) & (data[dep_var] == 1), 1, 0)

    # Condition for removing countries with population less than 1000 and nulls
    data = data[(data['avpop'] >= 1000) & (~data['avpop'].isnull())]

    # Forward fill by group all the null values in the regressors
    data[thetas] = data.groupby('countryid')[thetas].ffill()

    # Forward fill by group all the null token values
    data['tokens'] = data.groupby('countryid')['tokens'].ffill()

    if onset:  # Condition for which instances of the dependent variable to remove
        data = data[data[dep_var] != 1]  # Don't want repetition of conflict
    else:
        data = data[~data[dep_var].isnull()]

    # Only take data where tokens are non-zero and non-null
    data = data[(data['tokens'] > 0) & (~data['tokens'].isnull())]

    # If individuals with no conflict are to be removed
    if not all_indiv:
        total = data.groupby('countryid', as_index=False)[dep_var].sum()
        total = total.rename({dep_var: 'total_conflict'}, axis=1)
        data = data.merge(total, how='left', on='countryid')
        data = data[data['total_conflict'] > 0]

    # Remove all years after fit_year, authors define a sample variable in code
    data = data[data['year'] <= fit_year]

    data.set_index(['countryid', 'year'], inplace=True)

    regressors = thetas

    # If the interaction list is not empty, add the interactions
    if interactions is not None:
        for interact in interactions:
            cols = [x + "BY" + interact for x in thetas]
            data[cols] = data[thetas].multiply(data[interact], axis='index')
            regressors.extend(cols)

    # Add all of the independent regressors to the exog matrix
    exog = data[regressors]
    exog = sm.add_constant(exog)

    model = PanelOLS(data['one_before'], exog,
                     entity_effects=FE, time_effects=True)

    model = model.fit(cov_type='clustered', cluster_entity=True)

    return(model)


# 1b. Blundell Bond

def blundell_bond(data, params):

    ########################################################################
    # Input Notes
    ########################################################################
    # data should be master file
    # fit year is year t, before prediction year

    fit_year = params['fit_year']

    dep_var = params['dep_var']  # dep var should be bdbest25 or bdbest1000

    # If not all_indiv, then removes counties with no conflict in the sample
    all_indiv = params['all_indiv']

    # Get the list of variables to interact with the theta shares
    # This input diverges from the authors technique, so results will differ
    interactions = params['interactions']

    max_lags = params['max_lags'] - 1

    # The order of null filling and removal is done according to the authors
    #       code, changing the order changes the result
    ############################################################################
    # Input Notes
    ############################################################################

    data = data.copy(deep=True)  # Make a copy so dataframe not overwritten
    data = data[data['theta_year'] == (fit_year + 1)]

    # Define the column names of thetas to be used as regressors
    thetas = ["theta" + str(i) for i in range(1, 15)]

    # Forward fill by group all the null values in the regressors
    data[thetas] = data.groupby('countryid')[thetas].ffill()
    regressors = thetas
    regressors.append(dep_var)

    # If the interaction list is not empty, add the interactions
    if interactions is not None:
        for interact in interactions:
            cols = [x + "BY" + interact for x in thetas]
            data[cols] = data[thetas].multiply(data[interact], axis='index')
            regressors.extend(cols)

    # Forward fill by group all the null token values
    data['tokens'] = data.groupby('countryid')['tokens'].ffill()

    # Condition for removing countries with population less than 1000 and nulls
    data = data[(data['avpop'] >= 1000) & (~data['avpop'].isnull())]

    # Only take data where tokens are non-zero and non-null
    data = data[(data['tokens'] > 0) & (~data['tokens'].isnull())]

    # If individuals with no conflict are to be removed
    if not all_indiv:
        total = data.groupby('countryid', as_index=False)[dep_var].sum()
        total = total.rename({dep_var: 'total_conflict'}, axis=1)
        data = data.merge(total, how='left', on='countryid')
        data = data[data['total_conflict'] > 0]

    # Remove all years after fit_year, authors define a sample variable in code
    data = data[data['year'] <= fit_year]
    data = data[['countryid', 'year'] + regressors]
    data['year'] = data['year'] - data['year'].min() + 1
    max_year = int(data['year'].max())
    data['year'] = data['year'].apply(lambda x: 'Year' + str(int(x)))
    data = data.pivot(index='countryid', columns='year')[regressors]
    data.columns = [col[1] + "_" + col[0]
                    for col in data.columns.values]

    for i in range(1, max_year):
        for col in regressors:
            diff_col = 'Diff' + "_" + str(i + 1) + "_" + str(i) + "_" + col
            col1 = 'Year' + str(i + 1) + "_" + col
            col2 = 'Year' + str(i) + "_" + col
            data[diff_col] = data[col1] - data[col2]

    formula = dict()

    for i in range(1, (max_year - 1)):
        l_dep = 'Year' + str(i + 2) + "_" + dep_var
        d_dep = 'Diff_' + str(i + 2) + "_" + str(i + 1) + "_" + dep_var
        l_endog = ['Year' + str(i + 1) + "_" + col for col in regressors]
        d_endog = ['Diff_' + str(i + 1) + "_" + str(i) +
                   "_" + col for col in regressors]
        d_inst = ['Year' + str(j) + "_" + col for col in regressors
                  for j in range(max(1, i - max_lags), (i + 1))]
        l_inst = ['Diff_' + str(i + 1) + "_" + str(i) +
                  "_" + col for col in regressors]
        formula['level' + str(i)] = l_dep + " ~ [" + \
            " + ".join(l_endog) + " ~ " + " + ".join(l_inst) + "]"
        formula['diff' + str(i)] = d_dep + " ~ [" + \
            " + ".join(d_endog) + " ~ " + " + ".join(d_inst) + "]"

    mod = IVSystemGMM.from_formula(formula, data, weight_type='unadjusted')

    constraints = []
    params = mod.param_names
    row = 0
    used = []
    for col in params:
        var = col.split("_")[-1]
        lst = [i for i in params if (i.split("_")[-1] == var) & (col != i)]
        for col2 in lst:
            if col2.split("_")[-1] in used:
                continue
            constraints.append([0] * len(params))
            constraints[row][params.index(col)] = 1
            constraints[row][params.index(col2)] = -1
            row += 1
        used.append(col.split("_")[-1])

    constraints = pd.DataFrame(constraints)

    mod.add_constraints(r=constraints)
    fit = mod.fit(cov_type='robust', iter_limit=1000)

    return(fit)


# 2a. Use model for prediction

def pred_model(data, model, params):

    ############################################################################
    # Input Notes
    ############################################################################
    # data should be master file
    # model input should be an object returned by run_model function

    # fit year is year t, before prediction year
    fit_year = params['fit_year']

    include_fixef = params['FE']  # Boolean for including fixed effects alpha

    # Get the list of variables to interact with the theta shares
    # This input diverges from the authors technique, so results will differ
    interactions = params['interactions']
    ############################################################################
    # Input Notes
    ############################################################################

    # Create a copy to not overwrite the dataframe
    data = data.copy(deep=True)

    # We are making a prediction for t + 1, as in authors code
    data = data[(data['year'] == (fit_year + 1)) &
                (data['theta_year'] == (fit_year + 1))]

    # Get the index of the dataframe, individual by time
    data = data.set_index(['countryid', 'year'])

    # Get the columns for the theta regressors
    thetas = ["theta" + str(i) for i in range(1, 15)]

    regressors = thetas

    # If the interaction list is not empty, add the interactions
    if interactions is not None:
        for interact in interactions:
            cols = [x + "X" + interact for x in thetas]
            data[cols] = data[thetas].multiply(data[interact], axis='index')
            regressors.extend(cols)

    # Add all of the independent regressors to the exog matrix
    exog = data[regressors]
    exog = sm.add_constant(exog)

    # Use the model to predict the t + 1 values for the chosen conflict
    preds = model.predict(exog)
    preds.reset_index(inplace=True)  # Reset the index

    # Condition for whether to break the predictions down by fixed and within
    # The authors do this, getting the alpha and within portion of prediction
    if include_fixef:
        # Rename the prediction, this in the linear prediction component
        preds = preds.rename({'predictions': 'within_pred'}, axis=1)

        # Get the estimated fixed effects (alpha) per country
        fixef = model.estimated_effects

        fixef = fixef.reset_index()  # Reset the index

        # Groupby country to get a single value from the estimated effects
        fixef = fixef.groupby('countryid',
                              as_index=False)['estimated_effects'].mean()

        # Rename the estimated effects as the fixed effects (alpha)
        fixef = fixef.rename({'estimated_effects': "FE"}, axis=1)
        # Remerge the fixed effects back onto the prediction frame
        preds = preds.merge(fixef, how='left', on='countryid')

        # The overall prediction should be the within and fixed predictions
        preds['predictions'] = preds['within_pred'] + \
            np.where(preds['FE'].isnull(), 0, preds['FE'])

        # preds = preds.drop('FE', axis = 1)

    return(preds)

# 2b. Return ROC

def compute_roc(master, model_params, file):

    dep_var = model_params['dep_var']

    true = master[(master['theta_year'] == master['year']) &
                  (master['year'] >= 1996)][['countryid', 'year', model_params['dep_var']]]

    pred_dfs = []

    merge_cols = ['countryid', 'year', 'predictions']
    if model_params['FE']:
        merge_cols.append('within_pred')

    for fit_year in range(1995, 2013):

        model_params['fit_year'] = fit_year

        model = run_model(master, model_params)

        preds = pred_model(master, model, model_params)

        pred_dfs.append(preds[merge_cols])

    preds = pd.concat(pred_dfs)

    true = true.merge(preds, how='left', on=['countryid', 'year'])

    true = true.dropna(axis=0)

    overall = roc_curve(true[dep_var], true['predictions'])
    within = roc_curve(true[dep_var], true['within_pred'])

    overall_auc = roc_auc_score(true[dep_var], true['predictions'])
    within_auc = roc_auc_score(true[dep_var], true['within_pred'])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.plot(overall[0], overall[1], 'b',
             color='black', label='Overall Prediciton')
    plt.plot(within[0], within[1], 'b', color='blue',
             linestyle='dashed', label='Within Prediction')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    fig.suptitle("ROC Curve: " + ['Incidence', 'Onset'][model_params['onset']] +
                 " of " + all_labs['Labs'][model_params['dep_var']])

    plt.text(0.02, 0.07, r'Overall AUC = ' +
             str(round(overall_auc, 2)), fontsize=10)
    plt.text(0.02, 0.0, r'Within AUC = ' +
             str(round(within_auc, 2)), fontsize=10)

    fig.savefig(file)

# 3. Function to output latex regression table

def out_latex(models, labs, model_params, file, type):

    ############################################################################
    # Input Notes
    ############################################################################
    # model should be fit object returned by run_model function
    # labs should be the label dictionary defined near beginining of script
    # model_params should be dictionary used in run_model function
    # file should be the file path to write latex string to
    # If type is custom then implements latex table written by hand, else
    #   the latex table is the one constructed by python (a little busy)
    labs = labs.copy()
    lag_labs = labs['Lag_Labs']
    labs = labs['Labs']

    use_lags = model_params['lagged_regs']
    ############################################################################
    # Input Notes
    ############################################################################

    # Define the beginning of latex tabular to be put within table definition
    string = "\\begin{center}\n\\begin{tabular}{l" + "c" * len(models) + \
        "}\n" + "\\\\[-1.8ex]\\hline\n"

    string += "".join(["& $\\textbf{" + key + "}$" for key in models.keys()])
    string += "\\\\\n \\hline \\hline \n"
    string += "& \\multicolumn{" + str(len(models)) + "}{c}{" + \
        labs[model_params['dep_var']] + "} \\\\\n" + "\\hline \\\\[-1.8ex]\n"

    param_ind = models[list(models.keys())[0]].params.index

    indices = [i for i in range(len(param_ind)) if "BY" not in param_ind[i]]

    # Get coefficient values
    params = [models[key].params.values[indices] for key in models]
    # Get standard error values
    errors = [models[key].std_errors.values[indices] for key in models]
    # Get the pvalue values
    pvals = [models[key].pvalues.values[indices] for key in models]
    # List of Labels
    if not use_lags:
        lab_list = [labs[i] for i in param_ind[indices]]
    else:
        lab_list = [lag_labs[i.split("_")[-1]] for i in param_ind[indices]]

    if use_lags:
        num_params = len(set([i.split("_")[-1] for i in param_ind[indices]]))
        params = params[:num_params]
        errors = errors[:num_params]
        pvals = pvals[:num_params]
        lab_list = lab_list[:num_params]

    # Iterate through the number of coefficients
    for i in range(len(lab_list)):
        if i != 0:
            string += "\\\\"

        string += lab_list[i]

        for j in range(len(models)):
            string += " & " + str("{:.4f}".format(params[j][i]))

            # Include the p value stars depending on the value
            if pvals[j][i] <= 0.01:
                string += "$^{***}$"
            elif pvals[j][i] <= 0.05:
                string += "$^{**}$"
            elif pvals[j][i] <= 0.1:
                string += "$^{*}$"

        string += " \\\\\n "

        for j in range(len(models)):

            # Add the errors below the coefficients
            string += "& (" + str("{:.4f}".format(errors[j][i])) + ")"

        string += "\\\\ \n"

    # Include whether Time/Entity Effects were used
    string += "\\hline \\\\[-1.8ex]\n \\\\ Included Effects: "

    for model in models:
        if not use_lags:
            effects = models[model].included_effects
        else:
            effects = ['Time', 'Entity']
        string += " & " + ", ".join([effect for effect in effects])

    # Include R-Squared in the outputs
    string += "\\\\ R-Squared: "
    string += " & " + \
        "&".join(["%.3f" % model.rsquared
                  for model in models.values()])

    # Include number of observations
    string += "\\\\ Observations: "
    string += " & " + "&".join([str(model.nobs) for model in models.values()])

    string += "\\end{tabular}\n\\end{center}"

    string += "Cluster Robust Standard Errors"

    # After creating latex string, write to tex file
    with open(file, 'w') as f:
        f.write(string)