##########################################################################
# This script provides descriptive statistics and visual analysis of the
# theta loadings and the dependent variable.
##########################################################################

# cross import data
from DataPrep import *

# import librariesp
import plotly.express as px
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas
from statsmodels.tsa.stattools import acf, pacf, adfuller
import warnings

warnings.filterwarnings('ignore')

##########################
# Summary statistics
##########################

# Define the summary columns to display in the panel summary
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if 'theta' in col])
sum_cols.extend(interactions)

# Define the topic generation year to use for the summary, currently 2013
master2013 = master[master['theta_year'] == 2013][sum_cols]
master2013.reset_index(inplace=True)
master2013.drop('index', axis=1, inplace=True)


def xtsum(data, labs, indiv, time):

    data = data.copy(deep=True)
    # Don't need to summarize the index columns
    cols = [col for col in sum_cols if col not in [indiv, time]]

    # Create a set of indices correpsonding to the final output
    df = pd.DataFrame(product([labs[x] for x in cols],
                              ['overall', 'between', 'within']), columns=['Variable', 'Type'])

    # Create the set of columns that will be dislpayed in the summary
    df['Mean'], df['Std. Dev.'], df['Min'], df['Max'] = 1, 1, 1, 1
    df['Observations'] = 1
    df.set_index(['Variable', 'Type'], inplace=True)

    # Loop through the summary columns and run the correpsonding functions
    for col in cols:
        df.loc[(labs[col], 'overall'), 'Mean'] = np.mean(data[col])
        df.loc[(labs[col], 'overall'), 'Std. Dev.'] = np.std(data[col])
        df.loc[(labs[col], 'overall'), 'Min'] = np.min(data[col])
        df.loc[(labs[col], 'overall'), 'Max'] = np.max(data[col])
        df.loc[(labs[col], "overall"), 'Observations'] = data[col].count()

        df.loc[(labs[col], 'between'), 'Mean'] = np.nan
        df.loc[(labs[col], 'between'), 'Min'] = \
            np.min(data.groupby(time, as_index=False)[col].mean()[col])

        df.loc[(labs[col], 'between'), 'Max'] = \
            np.max(data.groupby(time, as_index=False)[col].mean()[col])

        df.loc[(labs[col], "between"), 'Observations'] = \
            len(data[~np.isnan(data[col])][indiv].unique())

        df.loc[(labs[col], 'between'), 'Std. Dev.'] = np.sqrt(np.sum(np.power(data.groupby(time)[
            col].mean() - np.mean(data[col]), 2)) / (df.loc[(labs[col], 'between'), 'Observations'] - 1))

        df.loc[(labs[col], 'within'), 'Mean'] = np.nan
        df.loc[(labs[col], 'within'), 'Min'] = \
            np.min(data[col] + np.mean(data[col]) - data.merge(
                data.groupby(indiv, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=[indiv], how='left')['mean'])

        df.loc[(labs[col], 'within'), 'Max'] = \
            np.max(data[col] + np.mean(data[col]) - data.merge(
                data.groupby(indiv, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=[indiv], how='left')['mean'])

        df.loc[(labs[col], "within"), 'Observations'] = \
            len(data[~np.isnan(data[col])][time].unique())

        df.loc[(labs[col], 'within'), 'Std. Dev.'] = \
            np.sqrt(np.sum(np.power(data[col] - data.merge(
                data.groupby(indiv, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=indiv, how='left')['mean'], 2))
                    / (data[col].count() - 1))

        df.fillna("", inplace=True)

    return(df)


df = xtsum(master2013, labs, 'countryid', 'year')
# Write the dataframe to latex with a few extra formatting add ons
df.to_latex(currDir + "/Report/xtsum_new.tex", bold_rows=True, multirow=True,
            float_format="{:0.3f}".format)

##########################
# Plots
##########################


# build data frame that allows for subsequent groupings
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if 'theta' in col])
sum_cols.extend(own)

master_plot = master[master['theta_year'] == 2013][sum_cols]
master_plot.reset_index(inplace=True)
master_plot.drop('index', axis=1, inplace=True)

topic_names = {'theta0': 'Industry', 'theta1': 'CivicLife1', 'theta2': 'Asia',
               'theta3': 'Sports', 'theta4': 'Justice', 'theta5': 'Tourism',
               'theta6': 'Politics', 'theta7': 'Conflict1',
               'theta8': 'Business', 'theta9': 'Economics',
               'theta10': 'InterRelations1',
               'theta11': 'InterRelations2', 'theta12': 'Conflict3',
               'theta13': 'CivicLife2', 'theta14': 'Conflict2'}

master_plot.rename(columns=topic_names, inplace=True)

country_names = FileRead('CountryID')

master_plot = master_plot.merge(country_names, on='countryid', how='left')


# limit countries to the ones that experienced both peace and conflict (check sd?)
conflict_std = master2013.groupby('countryid')[['bdbest25']].std()
conflict_countries = list(
    conflict_std[conflict_std['bdbest25'] != 0].index.values)

master_plot_conflict = master_plot[master_plot.countryid.isin(
    conflict_countries)]

# wide to long
master_plot_conflict_long = pd.melt(master_plot_conflict, id_vars=[
                                    'country', 'year'], value_vars=list(topic_names.values()))
master_plot_conflict_long = master_plot_conflict_long.merge(
    master_plot_conflict[['country', 'year', 'bdbest25', 'bdbest1000', 'region_o', 'subregion_o', 'discrimshare']], on=['country', 'year'], how='left')

# clean data (no values except for country, investigate this further!)
master_plot_conflict_long = master_plot_conflict_long[master_plot_conflict_long['region_o'] != '']


print("============================================================")
print("Creating and Saving Total Theta Plot in Report")
print("============================================================")

# plot joint theta development not distinguishing between countries
sns.lineplot(data=master_plot_conflict_long, x="year", y="value", hue="variable")
plt.savefig(currDir + str('/Report/thetas_total.png'))
plt.close()

print("============================================================")
print("Creating and Saving Theta per Country Plot in Report")
print("============================================================")

#  plot thetas for each country individually
fig, axes = plt.subplots(nrows=15, ncols=6)

for ax, country in zip(axes.flatten(), master_plot_conflict_long['country'].unique()):
    dat = master_plot_conflict_long[master_plot_conflict_long['country'] == country]
    conflict_years = dat[dat['bdbest25'] == 1].year.unique()
    for year in conflict_years:
        ax.axvline(year, color='red')
    sns.lineplot(data=dat, x="year", y="value", hue="variable", ax=ax)
    ax.set(title=str(country))
    ax.get_legend().remove()

handles, labels = fig.axes[-2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.set_size_inches(25, 25)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.suptitle('Distributions of thetas per country')
fig.savefig(currDir + str('/Report/thetas_perCountry.png'), dpi=100)
plt.close(fig)

print("============================================================")
print("Creating and Saving Theta per Region Plot in Report")
print("============================================================")

# replicate this exercise for regions

fig, axes = plt.subplots(nrows=3, ncols=2)

for ax, region in zip(axes.flatten(), master_plot_conflict_long['region_o'].unique()):
    dat = master_plot_conflict_long[master_plot_conflict_long['region_o'] == region]
    sns.lineplot(data=dat, x="year", y="value", hue="variable", ax=ax)
    ax.set(title=str(region))
    ax.get_legend().remove()

handles, labels = fig.axes[-2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.set_size_inches(18.5, 18.5)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.suptitle('Distributions of thetas per region')
fig.savefig(currDir + str('/Report/thetas_perRegion.png'), dpi=100)
plt.close(fig)


print("============================================================")
print("Creating and Saving Theta per Sub-Region Plot in Report")
print("============================================================")
# replicate this exercise for subregions

fig, axes = plt.subplots(nrows=6, ncols=3)

for ax, subregion in zip(axes.flatten(), master_plot_conflict_long['subregion_o'].unique()):
    dat = master_plot_conflict_long[master_plot_conflict_long['subregion_o'] == subregion]
    sns.lineplot(data=dat, x="year", y="value", hue="variable", ax=ax)
    ax.set(title=str(subregion))
    ax.get_legend().remove()

handles, labels = fig.axes[-2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.set_size_inches(18.5, 18.5)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.suptitle('Distributions of thetas per subregion')
fig.savefig(currDir + str('/Report/thetas_perSubregion.png'), dpi=100)
plt.close(fig)


print("============================================================")
print("Creating and Saving Theta vs. DiscrimShare Plot in Report")
print("============================================================")
# scatter of % discrimination vs theta, plot for each year
# no values for 2014

fig, axes = plt.subplots(nrows=8, ncols=5)

for ax, year in zip(axes.flatten(), master_plot_conflict_long[master_plot_conflict_long['year'] < 2014]['year'].unique()):
    dat = master_plot_conflict_long[master_plot_conflict_long['year'] == year]
    sns.scatterplot(data=dat, x="discrimshare", y="value", hue="variable", ax=ax)
    ax.set(title=str(year))
    ax.get_legend().remove()

handles, labels = fig.axes[-2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.set_size_inches(18.5, 18.5)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.suptitle('Thetas vs. share of discrimination per year')
fig.savefig(currDir + str('/Report/thetas_discrim.png'), dpi=100)
plt.close(fig)

print("============================================================")
print("Creating and Saving Conflict Map HTML in Report")
print("============================================================")

# conflict map to visualize variation in dependent variable
# interactive conflict map to visualize variation in dependent variable
master2013_map = master2013.groupby(
    ['countryid'])['bdbest25'].mean().reset_index()

master2013_map = master2013_map.merge(complete_file[['isocode', 'countryid', 'country']].drop_duplicates(subset=['isocode'], keep='first'),
                                      on='countryid', how='left')

fig = px.choropleth(master2013_map, locations="isocode",
                    color="bdbest25",  # lifeExp is a column of gapminder
                    hover_name="country",  # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Reds)

fig.write_html(currDir + '/Report/conflict_map.html')

print("============================================================")
print("Creating and Saving Static Conflict Map HTML in Report")
print("============================================================")

# static conflict map to visualize variation in dependent variable
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.columns = ['pop_est', 'continent',
                 'name', 'CODE', 'gdp_md_est', 'geometry']
conflict_map = pd.merge(world, master2013_map,
                        left_on='CODE', right_on='isocode')

conflict_map['coords'] = conflict_map['geometry'].apply(
    lambda x: x.representative_point().coords[:])
conflict_map['coords'] = [coords[0] for coords in conflict_map['coords']]


fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="3%", pad=0.1)

conflict_map.plot(column='bdbest25', legend=True, cmap='Reds', legend_kwds={'label': "Percentage of Years in Armed Conflict",
                                                                            'orientation': "horizontal"}, ax=ax, cax=cax)
# for idx, row in conflict_map.iterrows():
#     ax.annotate(s=row['isocode'], xy=row['coords'],
#                  horizontalalignment='center', size=2)
ax.set_axis_off()
plt.savefig(currDir + '/Report/conflict_map.png')
plt.close()

# Geting regression plots to show the most correlated interactions with
# Whether a country has ever experienced conflict

print("============================================================")
print("Creating and Saving Interaction vs. Conflict Plot in Report")
print("============================================================")

cols = ['childmortality', 'democracy', 'rgdpl', 'avegoodex']

df = master[master['theta_year'] == 2013]
df = df.groupby('countryid')[cols].mean().\
    melt(value_vars=cols, var_name='Interaction', ignore_index=False).\
    merge(df.groupby('countryid')['bdbest25'].agg(lambda x: np.where(
        x.sum() == 0, 0, 1)), how='left', left_index=True, right_index=True).\
    rename({'bdbest25': 'Conflict'}, axis=1)
df['Interaction'] = df['Interaction'].map(labs)

sns.set(style="whitegrid")
g = sns.FacetGrid(df, col='Interaction', sharex=False, sharey=False, col_wrap=2)
g.map(sns.regplot, 'value', 'Conflict', truncate=True, color='navy')
g.add_legend()
g.set(ylim=(-0.1, 1.1))
plt.savefig(currDir + "/Report/InteractionRegressions.png")
plt.close()

print("============================================================")
print("Creating and Saving Grouped ACF Plot in Report")
print("============================================================")

df = master[master['theta_year'] == 2013]
df = df.sort_values(by=['countryid', 'year'])


def get_acf(col, nlags):
    acfs = [0] * nlags
    lags = [str(i) for i in range(1, (nlags + 1))]
    count = 0

    for j, country in enumerate(df['countryid'].unique()):
        arr = acf(df[df['countryid'] == country]
                  [col], missing='drop', nlags=nlags)
        if any(pd.isna(arr)):
            pass
        else:
            acfs = [acfs[i] + arr[i] for i in range(len(acfs))]
            count += 1

    acfs = [i / count for i in acfs]
    acfs = pd.DataFrame([lags, acfs]).T
    acfs = acfs.rename({0: 'Lag', 1: 'ACF'}, axis=1)

    return(acfs)


fig, ax = plt.subplots(1)
plt.plot(get_acf('bdbest25', 20)['Lag'], get_acf('bdbest25', 20)['ACF'],
         label='Armed Conflict ACF')
plt.plot(get_acf('bdbest1000', 20)['Lag'], get_acf('bdbest1000', 20)['ACF'],
         label='Civil War ACF')
plt.plot(get_acf('bdbest', 20)['Lag'], get_acf('bdbest', 20)['ACF'],
         label='Battle Deaths ACF')
plt.legend(loc='upper right')
fig.suptitle('ACF of Dependent Variables')
ax.set_xlabel('Lag', fontsize=10)
ax.set_ylabel('ACF Value', fontsize=10)
plt.savefig(currDir + "/Report/DependentVar_AutoCorr.png")
plt.close()
