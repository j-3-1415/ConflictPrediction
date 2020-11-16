##########################################################################
# This script provides descriptive statistics and visual analysis of the
# theta loadings and the dependent variable.
##########################################################################

# cross import data
from Code.DataPrep import *

# import libraries
import plotly.express as px
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

##########################
# Summary statistics
##########################

# Define the summary columns to display in the panel summary
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if '_theta' in col])

# Define the topic generation year to use for the summary, currently 2013
master2013 = master[master['theta_year'] == 2013][sum_cols]
master2013.reset_index(inplace=True)
master2013.drop('index', axis=1, inplace=True)

def xtsum(data, labs, indiv, time):

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
            len(data[~np.isnan(data[col])][time].unique())

        df.loc[(labs[col], 'between'), 'Std. Dev.'] = np.sqrt(np.sum(np.power(data.groupby(time)[
            col].mean() - np.mean(data[col]), 2)) / (df.loc[(labs[col], 'between'), 'Observations'] - 1))

        df.loc[(labs[col], 'within'), 'Mean'] = np.nan
        df.loc[(labs[col], 'within'), 'Min'] = \
            np.min(data[col] + np.mean(data[col]) - data.merge(
                data.groupby(time, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=[time], how='left')['mean'])

        df.loc[(labs[col], 'within'), 'Max'] = \
            np.max(data[col] + np.mean(data[col]) - data.merge(
                data.groupby(time, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=[time], how='left')['mean'])

        df.loc[(labs[col], "within"), 'Observations'] = \
            len(data[~np.isnan(data[col])][indiv].unique())

        df.loc[(labs[col], 'within'), 'Std. Dev.'] = \
            np.sqrt(np.sum(np.power(data[col] - data.merge(
                data.groupby(time, as_index=False)[col].mean().rename(
                    {col: 'mean'}, axis=1), on=time, how='left')['mean'], 2))
                    / (data[col].count() - 1))

        df.fillna("", inplace=True)

        return(df)

##########################
# Plots
##########################

# build data frame that allows for subsequent groupings
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if '_theta' in col])
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


# plot joint theta development not distinguishing between countries
sns.lineplot(data=master_plot_conflict_long, x="year", y="value", hue="variable")
plt.savefig(currDir + str('/Report/thetas_total.png'))
plt.close()

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

# conflict map to visualize variation in dependent variable
master2013_map = master2013.groupby(
    ['countryid'])['bdbest25'].sum().reset_index()

master2013_map = master2013_map.merge(complete_file[['isocode', 'countryid', 'country']].drop_duplicates(subset=['isocode'], keep='first'),
                                      on='countryid', how='left')

fig = px.choropleth(master2013_map, locations="isocode",
                    color="bdbest25",  # lifeExp is a column of gapminder
                    hover_name="country",  # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Reds)

fig.write_html(currDir + '/Report/conflict_map.html')

