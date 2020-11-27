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
from DataPrep import *

# import libraries
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels import IVSystemGMM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# 1a. Fixed Effects model or Pooled OLS Model


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
    data = data[data['theta_year'] == fit_year]

    # Define the column names of thetas to be used as regressors
    thetas = params['topic_cols'].copy()

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
    # fit year is year before prediction year

    params = params.copy()
    fit_year = params['fit_year']

    dep_var = params['dep_var']  # dep var should be bdbest25 or bdbest1000

    # If not all_indiv, then removes counties with no conflict in the sample
    all_indiv = params['all_indiv']

    # Get the list of variables to interact with the theta shares
    # This input diverges from the authors technique, so results will differ
    interactions = params['interactions']

    max_lags = params['max_lags'] - 1
    maxldep = params['maxldep'] - 1

    depl = params['dep_lags']

    start = depl

    weight_type = params['weight_type']

    iters = params['iterations']

    cov_type = params['cov_type']

    # The order of null filling and removal is done according to the authors
    #       code, changing the order changes the result
    ############################################################################
    # Input Notes
    ############################################################################

    data = data.copy(deep=True)  # Make a copy so dataframe not overwritten
    data = data[data['theta_year'] == fit_year]
    data = data.sort_values(by=['countryid', 'year'])

    # Define the column names of thetas to be used as regressors
    thetas = params['topic_cols'].copy()
    data['L' + dep_var] = data[dep_var].copy()

    # Forward fill by group all the null values in the regressors
    data[thetas] = data.groupby('countryid')[thetas].ffill()
    if dep_var == 'bdbest':
        data[thetas] = data[thetas] * 100
    regressors = thetas.copy()
    regressors.append(dep_var)

    # If the interaction list is not empty, add the interactions
    if interactions is not None:
        for interact in interactions:
            data[interact] = data.groupby(
                'countryid')[interact].fillna(method='backfill')
            data[interact] = data.groupby(
                'countryid')[interact].fillna(method='ffill')
            if (interact in params['scaled']) & (dep_var != 'bdbest'):
                data[interact] = (data[interact] * 0.5) / data[interact].mean()
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
    data = data[['countryid', 'year'] + regressors + ['L' + dep_var]]
    data['year'] = data['year'] - data['year'].min() + 1
    max_year = int(data['year'].max())
    data['year'] = data['year'].apply(lambda x: 'Year' + str(int(x)))
    data = data.pivot(index='countryid', columns='year')[
        regressors + ['L' + dep_var]]

    data.columns = [col[1] + "_" + col[0]
                    for col in data.columns.values]

    for i in range(1, max_year):
        for col in regressors + ['L' + dep_var]:
            col1 = 'Year' + str(i + 1) + "_" + col
            col2 = 'Year' + str(i) + "_" + col
            diff_col = 'Diff_' + str(i + 1) + "_" + col
            data[diff_col] = data[col1] - data[col2]

    data = data.dropna(axis=0)

    formula = dict()

    for i in range(start, (max_year - depl)):
        l_dep = 'Year' + str(i + [1, 2][start == 1]) + "_" + dep_var

        d_dep = 'Diff_' + str(i + 2) + "_" + dep_var

        l_endog = ['Year' + str(i + [0, 1][start == 1]) +
                   "_" + col for col in regressors]

        d_endog = ['Diff_' + str(i + 1) + "_" + col for col in regressors]

        ldep_cols = [col for col in regressors if dep_var not in col]
        d_inst = ['Year' + str(j) + "_" + col for col in ldep_cols
                  for j in range(max(start, i - maxldep), (i + 1))]
        d_inst += ['Year' + str(j) + "_" + dep_var
                   for j in range(max(start, i - max_lags), (i + 1))]

        if depl >= 2:
            d_inst = ['Year' + str(j) + "_" + dep_var
                      for j in range(max(start - 1, i - max_lags - 1), (i))]
            d_inst += ['Year' + str(j) + "_" + col for col in ldep_cols
                       for j in range(max(start - 1, i - maxldep - 1), (i))]

        l_inst = ['Diff_' + str(i + [0, 1][start == 1]) +
                  "_" + col for col in regressors]

        formula['level' + str(i)] = l_dep + " ~ " + ["", 'Year' + str(i - depl + 1) + "_L" + dep_var + " + "][depl >= 2] + "[" + \
            " + ".join(l_endog) + " ~ " + " + ".join(l_inst) + "]"
        formula['diff' + str(i)] = d_dep + " ~ " + ["", 'Diff_' + str(i - depl + 2) + "_L" + dep_var + " + "][depl >= 2] + "[" + \
            " + ".join(d_endog) + " ~ " + " + ".join(d_inst) + "]"

    mod = IVSystemGMM.from_formula(formula, data, weight_type=weight_type)

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
    fit = mod.fit(cov_type=cov_type, iter_limit=iters)

    return(fit)


# 2a. Use model for prediction

def pred_model(data, model, params):

    ############################################################################
    # Input Notes
    ############################################################################
    # data should be master file
    # model input should be an object returned by run_model function
    params = params.copy()
    data = data.copy(deep=True)

    # Get the columns for the theta regressors
    thetas = params['topic_cols'].copy()
    ############################################################################
    # Input Notes
    ############################################################################

    # We are making a prediction for t + 1, as in authors code
    data = data[(data['theta_year'] == params['fit_year'])]

    # Get the index of the dataframe, individual by time
    data = data.set_index(['countryid', 'year'])

    regressors = thetas.copy()

    # If the interaction list is not empty, add the interactions
    if params['interactions'] is not None:
        for interact in params['interactions']:
            if interact in params['scaled']:
                data[interact] = (data[interact] * 0.5) / data[interact].mean()
            cols = [x + "BY" + interact for x in thetas]
            data[cols] = data[thetas].multiply(data[interact], axis='index')
            regressors.extend(cols)

    if not params['lagged_regs']:

        # Add all of the independent regressors to the exog matrix
        exog = data[regressors]
        exog = sm.add_constant(exog)
        exog = exog[exog.index.get_level_values(1) == params['fit_year']]

        # Use the model to predict the t + 1 values for the chosen conflict
        preds = model.predict(exog)
        preds.reset_index(inplace=True)  # Reset the index

        # Condition for whether to break the predictions down by fixed and within
        # The authors do this, getting the alpha and within portion of prediction
        if params['FE']:
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

    else:
        regressors.append(params['dep_var'])
        exog = data[regressors]
        exog = exog[exog.index.get_level_values(1) == params['fit_year']]
        if params['dep_lags'] >= 2:
            exog['L' + params['dep_var']] = exog.groupby(exog.index.get_level_values(0))[
                params['dep_var']].shift()

        indices = list(set([i.split("_")[-1] for i in model.params.index]))
        num_params = len(indices)

        mapper = dict(zip(indices, model.params[:num_params]))
        preds = exog.dot(pd.Series(mapper)).reset_index().\
            rename({0: 'within_pred'}, axis=1)
        preds['year'] = preds['year'] + 1

    return(preds)

# 2b. Return ROC


def compute_roc(master, model_params, file):

    master = master.copy(deep=True)
    params = model_params.copy()

    add_overall = (params['FE']) & (not params['lagged_regs'])

    if not params['lagged_regs']:
        if params['FE']:
            model_name = 'FE'
        else:
            model_name = 'Pooled OLS'
    else:
        model_name = 'Blundell-Bond'

    dep_var = params['dep_var']

    true = master[master['theta_year'] == master['year']
                  ][['countryid', 'year', params['dep_var']]]
    true = true.set_index(['countryid', 'year'])

    true = true[true.index.get_level_values(1) >= 1996.0]

    print(true.head())

    pred_dfs = [None] * len(range(1995, 2014))

    merge_cols = ['countryid', 'year', 'within_pred']
    if add_overall:
        merge_cols.append('predictions')

    pbar = tqdm(range(1995, 2014), leave=True)

    for fit_year in pbar:

        params['fit_year'] = fit_year

        if not params['lagged_regs']:
            model = run_model(master, params)
        else:
            model = blundell_bond(master, params)

        preds = pred_model(master, model, params)

        pred_dfs[fit_year - 1995] = preds[merge_cols]

    preds = pd.concat(pred_dfs).set_index(['countryid', 'year'])

    true = true.merge(preds, how='inner', left_index=True, right_index=True)

    true = true.dropna(axis=0)

    within = roc_curve(true[dep_var], true['within_pred'])
    within_auc = roc_auc_score(true[dep_var], true['within_pred'])

    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.plot(within[0], within[1], 'b', color='blue',
             linestyle='dashed', label='Within Prediction')
    plt.text(0.02, 0.0, r'Within AUC = ' +
             str(round(within_auc, 2)), fontsize=10)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.legend(loc='lower right')

    title = 'ROC Curve: ' + model_name
    if not params['lagged_regs']:
        title += [' Incidence', ' Onset'][model_params['onset']]
    title += ' of ' + all_labs['Labs'][params['dep_var']]

    fig.suptitle(title)

    if add_overall:

        overall = roc_curve(true[dep_var], true['predictions'])
        overall_auc = roc_auc_score(true[dep_var], true['predictions'])

        plt.plot(overall[0], overall[1], 'b',
                 color='black', label='Overall Prediciton')

        plt.text(0.02, 0.07, r'Overall AUC = ' +
                 str(round(overall_auc, 2)), fontsize=10)

    fig.savefig(file)


# Function to calculate the second order panel serial correlation significance
def estat(model, lags):
    resid = model.wresids.copy()
    resid = resid[[col for col in resid.columns if 'level' not in col]]
    resid.columns = pd.MultiIndex.from_tuples(
        [('diff', col[4:]) for col in resid.columns])
    resid = resid.stack(level=1).reset_index().rename(
        {'level_0': 'countryid', 'level_1': 'year'}, axis=1)
    resid['year'] = resid['year'].astype(int)
    resid['countryid'] = resid['countryid'].astype(int)

    resid = resid.sort_values(by=['countryid', 'year'])
    for lag in range(1, lags + 3):
        resid['AR' + str(lag)] = resid.groupby('countryid')['diff'].shift(lag)
    resid = resid.set_index(['countryid', 'year'])

    y = resid['diff']
    exog = resid[[col for col in resid.columns if "AR" + str(lags) in col]]
    exog = sm.add_constant(exog)

    res_mod = PanelOLS(y, exog, time_effects=True, entity_effects=True).fit(
        cov_type='clustered', cluster_entity=True)

    param = res_mod.params[list(
        res_mod.params.index).index('AR' + str(lags))]
    pvalue = res_mod.pvalues[list(
        res_mod.params.index).index('AR' + str(lags))]

    return(pvalue)


# 3. Function to output latex regression table
def out_latex(models, labs, model_params, file):

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

    order = [model_params['dep_var'], 'L' + model_params['dep_var']]
    theta_order = ['theta2', 'theta8', 'theta1', 'theta13', 'theta7', 'theta14', 'theta12', 'theta9', 'theta0', 'theta10', 'theta11', 'theta4',
                   'theta6', 'theta3', 'theta5']
    inter_order = ['Interacted' + col for col in theta_order if 'theta' in col]
    temp = [None] * (len(theta_order) * 2)
    temp[::2] = theta_order
    temp[1::2] = inter_order
    order.extend(temp)
    order.append('const')
    ############################################################################
    # Input Notes
    ############################################################################

    # Define the beginning of latex tabular to be put within table definition
    string = "\\renewcommand{\\arraystretch}{0.5}"

    string += "\\begin{center}\n\\begin{longtable}{l" + "c" * len(models) + \
        "}\n" + "\\\\[-1.8ex]\\hline\n"

    string += "& \\multicolumn{" + str(len(models)) + "}{c}{Dependent Variable = " + \
        labs[model_params['dep_var']] + "} \\\\\n" + "\\hline \\\\[-1.8ex]\n"
    string += "".join(["& $\\textbf{" + key + "}$" for key in models.keys()])
    string += "\\\\\n \\hline \\hline \n"

    full_ind = []
    for key in models.keys():
        indices = [
            "Interacted" + i.split("BY")[0] if 'BY' in i else i for i in models[key].params.index]
        if use_lags:
            indices = [
                "Interacted" + i.split("_")[-1].split("BY")[0] if 'BY' in i else i.split("_")[-1] for i in models[key].params.index]
        full_ind.extend(indices)

    full_ind = set(full_ind)
    full_ind = [col for col in order if col in full_ind]

    num_params = len(full_ind)

    vals = {key: {'params': {}, 'errors': {}, 'pvals': {}}
            for key in models.keys()}

    for key in models.keys():
        indices = [
            "Interacted" + i.split("BY")[0] if 'BY' in i else i for i in models[key].params.index]
        if use_lags:
            indices = [
                "Interacted" + i.split("_")[-1].split("BY")[0] if 'BY' in i else i.split("_")[-1] for i in models[key].params.index]
        params = models[key].params.values
        errors = models[key].std_errors.values
        pvals = models[key].pvalues.values

        num_params = len(set(indices))

        if use_lags:
            params = params[:num_params]
            errors = errors[:num_params]
            pvals = pvals[:num_params]

        vals[key]['params'] = dict(zip(indices, params))
        vals[key]['errors'] = dict(zip(indices, errors))
        vals[key]['pvals'] = dict(zip(indices, pvals))

    if not use_lags:
        lab_list = {i: labs[i] for i in full_ind}
    else:
        lab_list = {i: lag_labs[i] for i in full_ind}

    # Iterate through the number of coefficients
    for i, ind in enumerate(full_ind):
        string += "\\\\"

        string += lab_list[ind]

        for key in models.keys():
            if ind in vals[key]['params'].keys():
                param = str("{:.4f}".format(vals[key]['params'][ind]))
                if vals[key]['pvals'][ind] <= 0.01:
                    star = "$^{***}$"
                elif vals[key]['pvals'][ind] <= 0.05:
                    star = "$^{**}$"
                elif vals[key]['pvals'][ind] <= 0.1:
                    star = "$^{*}$"
                else:
                    star = ""
            else:
                param = ""
                star = ""

            string += " & " + param + star

        string += "\\\\ \n "

        for key in models.keys():

            if ind in vals[key]['params'].keys():
                error = "(" + \
                    str("{:.4f}".format(vals[key]['errors'][ind])) + ")"
            else:
                error = ""
            string += "&" + error

        string += "\n"

    # Include whether Time/Entity Effects were used
    string += "\\\\ \\hline \\\\[-1.8ex]\n Included Effects: "

    for model in models:
        if not use_lags:
            effects = models[model].included_effects
        else:
            effects = ['Time']
        string += " & " + ", ".join([effect for effect in effects])

    # Include R-Squared in the outputs
    string += "\\\\ R-Squared: "
    string += " & " + \
        "&".join(["%.3f" % model.rsquared
                  for model in models.values()])

    # Include number of observations
    string += "\\\\ Observations: "
    string += " & " + "&".join([str(model.nobs) for model in models.values()])
    if use_lags:
        string += "\\\\ Over-Identification p-Val: "
        string += " & " + \
            "&".join(["%.3f" % model.j_stat.pval for model in models.values()])
        string += "\\\\ AB AR Order " + \
            str(2) + " p-Val: "
        string += " & " + "&".\
            join(["%.3f" % estat(model, 2)
                  for model in models.values()])
        string += "\\\\ Iterations: "
        string += " & " + "&".join([str(model.iterations)
                                    for model in models.values()])

    string += "\\\\ \\hline \\\\ \n \\multicolumn{" + str(
        len(models)) + "}{c}{Robust Standard Errors are Shown in Parentheses} \\\\"

    if use_lags:
        string += "\n \\multicolumn{" + str(len(models)) + "}{c}{Max Lag Depth = " + str(
            model_params['max_lags']) + "} \\\\"

    string += "\\end{longtable}\n\\end{center}"

    # After creating latex string, write to tex file
    with open(file, 'w') as f:
        f.write(string)
    # After creating latex string, write to tex file
    with open(file, 'w') as f:
        f.write(string)
