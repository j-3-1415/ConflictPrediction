######################=======================================>>>>>>>>>>>>>>>>>>>
# Panel Data 2020 Replication Project
# Luca Poll, Jacob Pichelmann, Andrew Boomer
# Last Modified By: Andrew Boomer 09/10/2020
# Modifications: Writing skeleton of code, organizing file structure
######################=======================================>>>>>>>>>>>>>>>>>>>

################################################################################
######################>>>>>>>TABLE OF CONTENTS<<<<<<<<<#########################
################################################################################

	######################>>>>>>>SECTION 1<<<<<<<<<#############################

	# Library Importation and directory definitions

	######################>>>>>>>SECTION 2<<<<<<<<<#############################

	# Merging dependent variable and covariates with theta data
	# Replicating code in Step1 Do File for generating necessary model variables
	# Create Dictionary of Column Definitions

	######################>>>>>>>SECTION 3<<<<<<<<<#############################

	#Descriptive statistics of data condusive to panel analysis, 
	#		i.e. pulled from class
	#		Also graphical analysis
	#		Luca can do his simulation in this section?

	######################>>>>>>>SECTION 4<<<<<<<<<#############################

	#Replicating baseline model in research without covariates

	######################>>>>>>>SECTION 5<<<<<<<<<#############################

	# ¿Replication of Civil War and Armed conflict models and ROC curves?
	# Not yet assigned to group member

	######################>>>>>>>SECTION 6<<<<<<<<<#############################

	# ¿What to do next????

################################################################################
####################>>>>>>>END TABLE OF CONTENTS<<<<<<<<<#######################
################################################################################

################################################################################
###########################>>>>>>>SECTION 1<<<<<<<<<############################
##############>>>>>>>Importing Libaries and Define Files<<<<<<<<<###############
################################################################################

print("=======================================================================")
print("Beginning Running Code in Section 1")
print("=======================================================================")

#Import libraries needed for the script, initial set, will likely need more

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import os
import re
from linearmodels.panel import PooledOLS, BetweenOLS, RandomEffects
from linearmodels.panel import PanelOLS, compare
from itertools import product
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.express as px
from linearmodels.iv.results import compare
from collections import OrderedDict
import semopy

global currDir #Define global variable of current directory
global local #Define global variable of input for file source
global FileDict #Define global variable for file importation

currDir = os.getcwd() #Define current directory based on python script

#Get user input whether files should be imported locally or from Dropbox
local = input("""Import data Locally? 'True' or 'False': 
	If 'False' Then Comes from Dropbox Online: """)

#Create a dictionary of files paths for dropbox. Used in FileRead function
FileDict = {
	'Topics' : ["wz1azpy6teayyzr", "topics.csv"],
	'TopicID' : ["ofgtf9ltq9eb8vt", "TopicID.csv"],
	'CountryID' : ["6je8xdzg63krmw5", "countryids.dta"],
	'EventData' : ["b1aahhlke2gd1n6", "eventdata.dta"],
	'CompleteMain' : ["srgz7t7k5t3omws", "complete_main_new.dta"],
	'CompleteMain_OG' : ["qrud7n5ak4pcawk", "complete_main_new_original.dta"],
	'ArticleCount' : ["vbz5dh78myqaa32", "article_count.dta"],
	'Theta_1995' : ["xqqb03az8dh10xo"], 'Theta_1996' : ["kn0n9kuc9342wyi"],
	'Theta_1997' : ["el9ltiga8fnyhnf"], 'Theta_1998' : ["kt9zo86nsgug18c"],
	'Theta_1999' : ["3e90uysagsh5j6y"], 'Theta_2000' : ["yird2kmqyfx4qos"],
	'Theta_2001' : ["ogbvafkck6uci1f"], 'Theta_2002' : ["dxybjz2nmc2wr1g"],
	'Theta_2003' : ["47xseavooz97h3i"], 'Theta_2004' : ["uymak66cgwh3cqz"],
	'Theta_2005' : ["pkcpx7n5kiiwmab"], 'Theta_2006' : ["20w1xwm9g3xoqxv"],
	'Theta_2007' : ["xnseiqxoahbfmns"],	'Theta_2008' : ["bezx6p9wu0t8bxo"],
	'Theta_2009' : ["cmfei42pxlmt3ca"], 'Theta_2010' : ["na35rztvy187dqi"],
	'Theta_2011' : ["r7x0478sjos6six"], 'Theta_2012' : ["5y5on73oxb4fgpf"],
	'Theta_2013' : ["of7mo0usbkqj82w"],	'Theta_2014' : ["vfe2xplgd2c9jw3"],
	'Theta_2015' : ["fqdq5ax5sixy8om"],
	'Theta_Full_1995' : ["z59vqlbrprczvnh"], 
	'Theta_Full_1996' : ["zlhr25gr9tlcycg"],
	'Theta_Full_1997' : ["nz8fqk6hh61f3v2"],
	'Theta_Full_1998' : ["ej30ikb0b3sh88e"],
	'Theta_Full_1999' : ["qqo39u38dqcqsve"],
	'Theta_Full_2000' : ["5g1v084txwt17yb"],
	'Theta_Full_2001' : ["wyh83n76aizguxr"],
	'Theta_Full_2002' : ["11oq0swh40sk93q"],
	'Theta_Full_2003' : ["v6x25dai1vbb481"],
	'Theta_Full_2004' : ["mp0470zj7gag5de"],
	'Theta_Full_2005' : ["ppcngl2qd3c53qq"],
	'Theta_Full_2006' : ["q3n2p8j4y0iocka"],
	'Theta_Full_2007' : ["u1z5j6d6v69dakh"],
	'Theta_Full_2008' : ["zyk9x1ti0tc7ry5"],
	'Theta_Full_2009' : ["z4rqncmntabs8iy"],
	'Theta_Full_2010' : ["9yja3dqqw3rquis"],
	'Theta_Full_2011' : ["4wnvqufctpvl1ms"],
	'Theta_Full_2012' : ["p5bfd51fbwya7d4"],
	'Theta_Full_2013' : ["j3ik866cvobs4rh"],
	'Theta_Full_2014' : ["fr5mo5raw1a6s5a"],
	'Theta_Full_2015' : ["3yit1p2sr3i7m0q"]}

#Create a function to read in the files based on codes within dictionaary
def FileRead(KeyName):

	#Split the dictionary key into a list
	KeyList = KeyName.split("_")

	#Condition to put together theta file name if needed
	if KeyList[0] == "Theta":
		FileName = "thetas15_alpha3_beta001_all_both"
		if KeyList[1] == "Full":
			FileName += KeyList[2] + ".dta"
		else:
			FileName += "_collapsed" + KeyList[1] + ".dta"
	else:
		FileName = FileDict[KeyName][1]

	#Condition for whether imported locally or from dropbox
	if local:
		path = currDir + "/dataverse_files/data/" + FileName
	else:
		code = FileDict[KeyName][0] #Get dropbox code
		path = "https://www.dropbox.com/s/" + code + "/" + FileName + "?dl=1"

	#DTA files are STATA files and need a different import function
	if '.dta' in path:
		return(pd.read_stata(path))
	else:
		return(pd.read_csv(path))

#Create a dictionary with labels for each column
labs = {'year' : 'Article Year', 'theta_year' : 'Topic Year', 
	'countryid' : 'CountryID', 'pop' : 'Country Population',
	'ste_theta0' : 'Topic 1 Share', 'ste_theta1' : 'Topic 2 Share',
	'ste_theta2' : 'Topic 3 Share', 'ste_theta3' : 'Topic 4 Share',
	'ste_theta4' : 'Topic 5 Share', 'ste_theta5' : 'Topic 6 Share',
	'ste_theta6' : 'Topic 7 Share', 'ste_theta7' : 'Topic 8 Share',
	'ste_theta8' : 'Topic 9 Share', 'ste_theta9' : 'Topic 10 Share',
	'ste_theta10' : 'Topic 11 Share', 'ste_theta11' : 'Topic 12 Share',
	'ste_theta12' : 'Topic 13 Share', 'ste_theta13' : 'Topic 14 Share',
	'ste_theta14' : 'Topic 15 Share', 'bdbest1000' : "Civil War",
	'bdbest25' : "Armed Conflict", "autoc" : "Autocracy", 
	"democ" : 'Democracy', 'const' : 'Constant'}

print("=======================================================================")
print("Finished Running Code in Section 1")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 1<<<<<<<<<##########################
################################################################################

################################################################################
###########################>>>>>>>SECTION 2<<<<<<<<<############################
########################>>>>>>>>Data Merging<<<<<<<#############################
################################################################################

print("=======================================================================")
print("Beginning Running Code in Section 2")
print("=======================================================================")

#Two main files needed for merge: thetas and complete data including dep var and controls

# get an overview of available data for controls
complete_file = FileRead("CompleteMain")
for col in complete_file.columns:
	print(col)

# specify range of years you want to use
theta_years = list(range(1995, 2015))

# specify the controls you want to use
own = ['region_o', 'subregion_o', 'discrimshare']

#Include the desired interaction variables to be include in master dataframe
interactions = ['autoc']
own.extend(interactions)

labs.update({"ste_theta" + str(i) + "X" + inter : \
	labs["ste_theta" + str(i)] + " by " + labs[inter] \
	for i in range(0, 15) for inter in interactions})

# in the paper they specify different sets of controls
# control_sets = {
# 	'chadefaux': ['cha_count', 'total_count', 'total_count_count', 'polity2', 
# 		'sincelast', 'sincelast_sq', 'sincelast_cu'],
# 	'chadefauxshort': ['cha_count', 'total_count', 'total_count_count'],
# 	'ajps': ['democ_3', 'democ_4', 'democ_5', 'democ_6', 'lnchildmortality', 'armedconf4', 'discrimshare'],
# 	'lasso': ['democ_3', 'democ_4', 'democ_6', 'childmortality', 'discrimshare', 'excludedshare', 'dissgov', 'ethnicgov'],
# 	'ward': ['high', 'low', 'autoc', 'democ', 'excludedshare', 'excludedshare_sq', 'lnchildmortality',  'armedconf4', 'discrimshare'],
# 	'events': ['govopp', 'dissgov', 'domgov', 'ethnicgov'],
# 	'pillars': ['nat_dum', 'nat_dum_goodinst',  'scmem', 'scmem_goodinst',  'scmem_cold', 'scmem_cold_goodinst']
# }

# get user input which set of controls to use
# print("""Available sets of controls:
# - chadefaux: cha_count total_count total_count_count polity2 sincelast sincelast_sq sincelast_cu
# - chadefauxshort: cha_count total_count total_count_count
# - ajps: democ_3 democ_4 democ_5 democ_6 lnchildmortality  armedconf4 discrimshare
# - lasso: democ_3 democ_4 democ_6 childmortality discrimshare excludedshare dissgov ethnicgov
# - ward: high low autoc democ excludedshare excludedshare_sq lnchildmortality  armedconf4 discrimshare
# - wardzwei: highdum lowdum autoc democ excludedshare excludedshare_sq lnchildmortality  armedconf4 discrimshare
# - events: govopp dissgov domgov ethnicgov
# - pillars: nat_dum nat_dum_goodinst  scmem scmem_goodinst  scmem_cold scmem_cold_goodinst
# """)
# controls_choice = input("Please choose 'own' if you specified your own controls or choose instead one of the sets above: ").lower()

# if controls_choice != 'own':
	# controls = control_sets[controls_choice]
# else:
	# controls = own

# list to store the ready to go regression data for each year
master_data = []

#Define index columns to merge on
merge_cols = ['year', 'countryid']
#Define the variables needed to construct the dependent variables
covars = ['bdbest25', 'bdbest1000', 'pop']
covars.extend(own)

for year in theta_years:
	# get thetas in the correct shape, i.e. keep only ste + year and countryid
	theta_file = FileRead("Theta_" + str(year))
	filter_col = [col for col in theta_file if col.startswith('ste')]
	filter_col.extend(merge_cols)
	filter_col.append('tokens')
	theta_file_sub = theta_file[filter_col]
	# controls have been specified outside of loop but we add merge keys here

	filter_col = []
	filter_col.extend(merge_cols)
	filter_col.extend(covars)
	complete_file_sub = complete_file.loc[:, complete_file.columns.isin(filter_col)]
	# recode dependent variables

	# merge thetas with the rest of the data
	merged = pd.merge(complete_file_sub, theta_file_sub, 
		how = 'left', on = ['year', 'countryid'])
	merged['theta_year'] = year
	master_data.append(merged)


master = pd.concat(master_data)
master = master.sort_values(by = ['theta_year', 'countryid', 'year'])
master2 = master.groupby('countryid', as_index = False)['pop'].mean()
master2.rename({'pop' : 'avpop'}, axis = 1, inplace = True)
master = master.merge(master2, on = 'countryid', how = 'left')
master = master[master['year'] > 1974]
master = master.reset_index().drop('index', axis = 1)

master.to_csv("STATATest.csv", index = False)
master.to_stata("STATATest.dta")

print("=======================================================================")
print("Finished Running Code in Section 2")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 2<<<<<<<<<##########################
################################################################################

################################################################################
###########################>>>>>>>SECTION 3<<<<<<<<<############################
#####################>>>>>>>>Descriptive Statistics<<<<<<<######################
################################################################################\

print("=======================================================================")
print("Beginning Running Code in Section 3")
print("=======================================================================")

#Define the summary columns to display in the panel summary
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if '_theta' in col])

#Define the topic generation year to use for the summary, currently 2013
master2013 = master[master['theta_year'] == 2013][sum_cols]
master2013.reset_index(inplace = True)
master2013.drop('index', axis = 1, inplace = True)


#Python doesn't have an xtsum function, so define one
def xtsum(data, labs, indiv, time):

	#Don't need to summarize the index columns
	cols = [col for col in sum_cols if col not in [indiv, time]]

	#Create a set of indices correpsonding to the final output
	df = pd.DataFrame(product([labs[x] for x in cols], 
		['overall', 'between', 'within']), columns = ['Variable', 'Type'])

	#Create the set of columns that will be dislpayed in the summary
	df['Mean'], df['Std. Dev.'], df['Min'], df['Max'] = 1, 1, 1, 1 
	df['Observations'] = 1
	df.set_index(['Variable', 'Type'], inplace = True)

	#Loop through the summary columns and run the correpsonding functions
	for col in cols:
		df.loc[(labs[col], 'overall'), 'Mean'] = np.mean(data[col])
		df.loc[(labs[col], 'overall'), 'Std. Dev.'] = np.std(data[col])
		df.loc[(labs[col], 'overall'), 'Min'] = np.min(data[col])
		df.loc[(labs[col], 'overall'), 'Max'] = np.max(data[col])
		df.loc[(labs[col], "overall"), 'Observations'] = data[col].count()

		df.loc[(labs[col], 'between'), 'Mean'] = np.nan
		df.loc[(labs[col], 'between'), 'Min'] = \
			np.min(data.groupby(time, as_index = False)[col].mean()[col])

		df.loc[(labs[col], 'between'), 'Max'] = \
			np.max(data.groupby(time, as_index = False)[col].mean()[col])

		df.loc[(labs[col], "between"), 'Observations'] = \
			len(data[~np.isnan(data[col])][time].unique())

		df.loc[(labs[col], 'between'), 'Std. Dev.'] = \
			np.sqrt(np.sum(np.power(data.groupby(time)[col].mean() - \
			np.mean(data[col]), 2)) / (df.loc[(labs[col], 'between'), \
			'Observations'] - 1))

		df.loc[(labs[col], 'within'), 'Mean'] = np.nan
		df.loc[(labs[col], 'within'), 'Min'] = \
			np.min(data[col] + np.mean(data[col]) - data.merge(\
			data.groupby(time, as_index = False)[col].mean().rename(\
				{col : 'mean'}, axis = 1), on = [time], how = 'left')['mean'])

		df.loc[(labs[col], 'within'), 'Max'] = \
			np.max(data[col] + np.mean(data[col]) - data.merge(\
				data.groupby(time, as_index = False)[col].mean().rename(\
				{col : 'mean'}, axis = 1), on = [time], how = 'left')['mean'])

		df.loc[(labs[col], "within"), 'Observations'] = \
			len(data[~np.isnan(data[col])][indiv].unique())

		df.loc[(labs[col], 'within'), 'Std. Dev.'] = \
			np.sqrt(np.sum(np.power(data[col] - data.merge(\
			data.groupby(time, as_index = False)[col].mean().rename(\
			{col : 'mean'}, axis = 1), on = time, how = 'left')['mean'], 2)) \
			/ (data[col].count() - 1))

		df.fillna("", inplace = True)

		return(df)

# return the summary dataframe
# df = xtsum(master2013, labs, 'countryid', 'year')
# Write the dataframe to latex with a few extra formatting add ons
# df.to_latex("xtsum.tex", bold_rows = True, multirow = True,
#	float_format = "{:0.3f}".format)

################################################################################
###########################>>>>>>>SECTION 3.5<<<<<<<<<##########################
#####################>>>>>>>>Visual Analysis<<<<<<<#############################
################################################################################

# build data frame that allows for subsequent groupings
sum_cols = ['year', 'countryid', 'bdbest25', 'bdbest1000']
sum_cols.extend([col for col in master.columns if '_theta' in col])
sum_cols.extend(own)

master_plot = master[master['theta_year'] == 2013][sum_cols]
master_plot.reset_index(inplace=True)
master_plot.drop('index', axis=1, inplace=True)

topic_names = {'ste_theta0': 'Industry', 'ste_theta1': 'CivicLife1', 'ste_theta2': 'Asia',
			   'ste_theta3': 'Sports', 'ste_theta4': 'Justice', 'ste_theta5': 'Tourism',
			   'ste_theta6': 'Politics', 'ste_theta7': 'Conflict1', 'ste_theta8': 'Business',
			   'ste_theta9': 'Economics', 'ste_theta10': 'InterRelations1', 'ste_theta11': 'InterRelations2',
			   'ste_theta12': 'Conflict3', 'ste_theta13': 'CivicLife2', 'ste_theta14': 'Conflict2'}

master_plot.rename(columns=topic_names, inplace=True)

country_names = FileRead('CountryID')

master_plot = master_plot.merge(country_names, on='countryid', how='left')


# limit countries to the ones that experienced both peace and conflict (check sd?)
conflict_std = master2013.groupby('countryid')[['bdbest25']].std()
conflict_countries = list(conflict_std[conflict_std['bdbest25'] != 0].index.values)

master_plot_conflict = master_plot[master_plot.countryid.isin(conflict_countries)]

# wide to long
master_plot_conflict_long = pd.melt(master_plot_conflict, id_vars=['country', 'year'], value_vars=list(topic_names.values()))
master_plot_conflict_long = master_plot_conflict_long.merge(
    master_plot_conflict[['country', 'year', 'bdbest25', 'bdbest1000', 'region_o', 'subregion_o', 'discrimshare']]
    , on=['country', 'year'], how='left')

# clean data (no values except for country, investigate this further!)
master_plot_conflict_long = master_plot_conflict_long[master_plot_conflict_long['region_o'] != '']


# plot joint theta development not distinguishing between countries
sns.lineplot(data=master_plot_conflict_long, x="year", y="value", hue="variable")
plt.savefig(currDir + str('/Output/thetas_total.png'))
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
fig.savefig(currDir + str('/Output/thetas_perCountry.png'), dpi=100)
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
fig.savefig(currDir + str('/Output/thetas_perRegion.png'), dpi=100)
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
fig.savefig(currDir + str('/Output/thetas_perSubregion.png'), dpi=100)
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
fig.savefig(currDir + str('/Output/thetas_discrim.png'), dpi=100)
plt.close(fig)

# conflict map to visualize variation in dependent variable
master2013_map = master2013.groupby(['countryid'])['bdbest25'].sum().reset_index()

master2013_map = master2013_map.merge(complete_file[['isocode', 'countryid', 'country']].drop_duplicates(subset=['isocode'], keep='first'),
                                      on='countryid', how='left')

fig = px.choropleth(master2013_map, locations="isocode",
                    color="bdbest25",  # lifeExp is a column of gapminder
                    hover_name="country",  # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Reds)

fig.write_html(currDir + '/Output/conflict_map.html')


print("=======================================================================")
print("Finished Running Code in Section 3")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 3<<<<<<<<<##########################
################################################################################

################################################################################
###########################>>>>>>>SECTION 4<<<<<<<<<############################
########################>>>>>>>>Initial Model<<<<<<<############################
################################################################################\

print("=======================================================================")
print("Beginning Running Code in Section 4")
print("=======================================================================")

# def data_compare(data, test, fit_year, compare_vars, dep_var, onset = True):

# 	data = data.copy()
# 	test = test.copy()
# 	thetas = [col for col in data.columns if \
# 		('ste_theta' in col)&(col != 'ste_theta0')]
	
# 	compare_vars = thetas + compare_vars

# 	test = test[(test['year'] > 1974)&(test['year'] <= fit_year)&
# 		(test['avpop'] >= 1000)&(test['tokens'] > 0)&(~test['avpop'].isnull())&
# 		(~test['tokens'].isnull())&(test['samp'] == 1)]

# 	if onset:
# 		test = test[test[dep_var] != 1]
# 	else:
# 		test = test[~test[dep_var].isnull()]

# 	test = test[['countryid', 'year'] + compare_vars]
# 	test = test.reset_index()
# 	data = data.reset_index()

# 	print('The number of rows of the data from STATA: ' + str(test.shape[0]))
# 	print('The number of rows of the data from Python: ' + str(data.shape[0]))

# 	for var in compare_vars:

# 		equality = all(data[var].values == test[var].values)
# 		print('For the variable ' + str(var) + 
# 			', the columns in STATA and Python are equal: ' + str(equality))
# 		if not equality:
# 			print(data[data[var] != test[var]][['countryid', 'year', var]].head())
# 			print(test[data[var] != test[var]][['countryid', 'year', var]].head())

#Define a function to run a model for a single theta year with inputs

def run_model(data, params):

	############################################################################
	#Input Notes
	############################################################################	
	#data should be master file
	#fit year is year t, before prediction year
	fit_year = params['fit_year']

	dep_var = params['dep_var'] #dep var should be bdbest25 or bdbest1000
	
	#onset is when there is no conflict in t but there is in t + 1
	#incidence is when there is conflict in t and conflict in t + 1
	onset = params['onset']

	#If not all_indiv, then removes counties with no conflict in the sample 
	all_indiv = params['all_indiv']

	FE = params['FE'] #Boolean for Pooled OLS or Fixed Effects Panel

	#Get the list of variables to interact with the theta shares
	#This input diverges from the authors technique, so results will differ
	interactions = params['interactions']

	Pooled = params['Pooled'] #Whether to run Pooled or not

	#The order of null filling and removal is done according to the authors
	#		code, changing the order changes the result
	############################################################################
	#Input Notes
	############################################################################

	#Get list of countries that are excluded for each run
	#Make a regression summary text file with each run

	data = data.copy(deep = True) #Make a copy so dataframe not overwritten
	data = data[data['theta_year'] == (fit_year + 1)]

	#Define the column names of thetas to be used as regressors
	thetas = ["ste_theta" + str(i) for i in range(1, 15)]

	#One before here means you are in t (one before the next period)
	data['one_before'] = data.groupby('countryid')[dep_var].shift(-1)

	if onset: #Condition for whether this is onset or incidence, explained above
		data['one_before'] = np.where(\
			(data['one_before'] == 1)&(data[dep_var] == 0), 1, 0)
	else:
		data['one_before'] = np.where(\
			(data['one_before'] == 1)&(data[dep_var] == 1), 1, 0)

	#Condition for removing countries with population less than 1000 and nulls
	data = data[(data['avpop'] >= 1000)&(~data['avpop'].isnull())]

	#Forward fill by group all the null values in the regressors
	data[thetas] = data.groupby('countryid')[thetas].ffill()

	#Forward fill by group all the null token values
	data['tokens'] = data.groupby('countryid')['tokens'].ffill()

	if onset: #Condition for which instances of the dependent variable to remove
		data = data[data[dep_var] != 1] #Don't want repetition of conflict
	else:
		data = data[~data[dep_var].isnull()]

	#Only take data where tokens are non-zero and non-null
	data = data[(data['tokens'] > 0)&(~data['tokens'].isnull())]

	#If individuals with no conflict are to be removed
	if not all_indiv:
		total = data.groupby('countryid', as_index = False)[dep_var].sum()
		total = total.rename({dep_var : 'total_conflict'}, axis = 1)
		data = data.merge(total, how = 'left', on = 'countryid')
		data = data[data['total_conflict'] > 0]

	# if Pooled:
	# 	for year in data['year'].unique()[1:-2]:
	# 		data['Dummy_' + str(year)] = np.where(data['year'] == year, 1, 0)

	#Remove all years after fit_year, authors define a sample variable in code
	data = data[data['year'] <= fit_year]
	data.set_index(['countryid', 'year'], inplace = True)

	regressors = thetas

	#If the interaction list is not empty, add the interactions
	if interactions is not None:
		for interact in interactions:
			cols = [x + "X" + interact for x in thetas]
			data[cols] = data[thetas].multiply(data[interact], axis = 'index')
			regressors.extend(cols)

	# if Pooled:
	# 	regressors.extend([x for x in data.columns if x.startswith("Dummy_")])

	#Add all of the independent regressors to the exog matrix
	exog = data[regressors]
	exog = sm.add_constant(exog)

	if Pooled:
		model = PooledOLS(data['one_before'], exog)
	else:
		model = PanelOLS(data['one_before'], exog,
			entity_effects = FE, time_effects = True)

	model = model.fit(cov_type = 'clustered', cluster_entity = True)

	return(model)

def pred_model(data, model, params):

	############################################################################
	#Input Notes
	############################################################################	
	#data should be master file
	#model input should be an object returned by run_model function

	#fit year is year t, before prediction year
	fit_year = params['fit_year']

	include_fixef = params['FE'] #Boolean for including fixed effects alpha

	#Get the list of variables to interact with the theta shares
	#This input diverges from the authors technique, so results will differ
	interactions = params['interactions']
	############################################################################
	#Input Notes
	############################################################################

	#Create a copy to not overwrite the dataframe
	data = data.copy(deep = True)

	#We are making a prediction for t + 1, as in authors code
	data = data[(data['year'] == (fit_year + 1))&\
		(data['theta_year'] == (fit_year + 1))]

	#Get the index of the dataframe, individual by time
	data = data.set_index(['countryid', 'year'])

	#Get the columns for the theta regressors
	thetas = ["ste_theta" + str(i) for i in range(1, 15)]

	regressors = thetas

	#If the interaction list is not empty, add the interactions
	if interactions is not None:
		for interact in interactions:
			cols = [x + "X" + interact for x in thetas]
			data[cols] = data[thetas].multiply(data[interact], axis = 'index')
			regressors.extend(cols)

	#Add all of the independent regressors to the exog matrix
	exog = data[regressors]
	exog = sm.add_constant(exog)

	#Use the model to predict the t + 1 values for the chosen conflict
	preds = model.predict(exog)
	preds.reset_index(inplace = True) #Reset the index

	#Condition for whether to break the predictions down by fixed and within
	#The authors do this, getting the alpha and within portion of prediction
	if include_fixef:
		#Rename the prediction, this in the linear prediction component
		preds = preds.rename({'predictions' : 'within_pred'}, axis = 1)

		#Get the estimated fixed effects (alpha) per country
		fixef = model.estimated_effects

		fixef = fixef.reset_index() #Reset the index

		#Groupby country to get a single value from the estimated effects
		fixef = fixef.groupby('countryid', 
			as_index = False)['estimated_effects'].mean()

		#Rename the estimated effects as the fixed effects (alpha)
		fixef = fixef.rename({'estimated_effects' : "FE"}, axis = 1)
		#Remerge the fixed effects back onto the prediction frame
		preds = preds.merge(fixef, how = 'left', on = 'countryid')

		#The overall prediction should be the within and fixed predictions
		preds['predictions'] = preds['within_pred'] + \
			np.where(preds['FE'].isnull(), 0, preds['FE'])

		# preds = preds.drop('FE', axis = 1)

	return(preds)

#Define function to output latex regression table
def out_latex(model, labs, model_params, file, type):

	############################################################################
	#Input Notes
	############################################################################	
	#model should be fit object returned by run_model function
	#labs should be the label dictionary defined near beginining of script
	#model_params should be dictionary used in run_model function
	#file should be the file path to write latex string to
	#If type is custom then implements latex table written by hand, else
	#	the latex table is the one constructed by python (a little busy)
	############################################################################
	#Input Notes
	############################################################################

	if type == 'custom': #Condition to create approximation of stargazer output

		params = model.params.values #Get coefficient values
		errors = model.std_errors.values #Get standard error values
		pvals = model.pvalues.values #Get the pvalue values
		lab_list = [labs[i] for i in model.params.index] #List of Labels

		#Define the beginning of latex tabular to be put within table definition
		string = "\\begin{center}\n\\begin{tabular}{lc}\n" + \
			"\\\\[-1.8ex]\\hline\n" + \
			"& $\\textbf{" + model_params['dep_var'] + "}$ \\\\\n" + \
			"\\hline \\\\[-1.8ex]\n"

		#Iterate through the number of coefficients
		for i in range(len(lab_list)):
			string += lab_list[i] + " & " + str("{:.4f}".format(params[i]))

			#Include the p value stars depending on the value
			if pvals[i] <= 0.01:
				string += "$^{***}$"
			elif pvals[i] <= 0.05:
				string += "$^{**}$"
			elif pvals[i] <= 0.1:
				string += "$^{*}$"

			#Add the errors below the coefficients
			string += " \\\\\n & (" + str("{:.4f}".format(errors[i])) + \
				") \\\\ \n"

		#Finish the tabular definition
		string += "\\hline \\\\[-1.8ex]\n\\end{tabular}\n\\end{center}"

		#Include notes for included effects and for robust standard errors
		string += "\nIncluded Effects: Time" + \
			["", ", Entity"][model_params["FE"]]
		string += "\nCluster Robust Standard Errors Used"

	else: #Condition for if using base PanelOLS summary
		string = model.summary.as_latex()

		#Loop through the coefficient names to replace with the labels
		for key in model.params.index:

			pattern = "textbf{" + key.replace("_", "\_") + "}"
			replace = "textbf{" + labs[key] + "}"
			string = string.replace(pattern, replace)

	return(string)

	#After creating latex string, write to tex file
	with open(file, 'w') as f:
		f.write(string)


# compare_vars = ['tokens', 'bdbest25', 'one_before', 'avpop']

# test = pd.read_stata(os.getcwd() + '/dataverse_files/data/test.dta')

model_params = {
	"fit_year" : 2013, #Year for fitting the model
	"dep_var" : "bdbest25", #Civil War (1000) or Armed Conflict (25)
	"onset" : True, #Onset of Incidence of Conflict
	"all_indiv" : True, #Include all countries or not
	"FE" : True, #Pooled Model or Fixed Effects
	'interactions' : None, #Set of interaction vars (can be None)
	'Pooled' : False
}

model = run_model(master, model_params)
print(model)

# data_compare(df, test, 1995, compare_vars, 'bdbest25', onset = True)

preds = pred_model(master, model, model_params)
print(preds)

string = out_latex(model, labs, model_params, "FE_1995.tex", "custom")

print("=======================================================================")
print("Finished Running Code in Section 4")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 4<<<<<<<<<##########################
################################################################################

################################################################################
###########################>>>>>>>SECTION 5<<<<<<<<<############################
#######################>>>>>>>>ROC Computation<<<<<<<###########################
################################################################################\

print("=======================================================================")
print("Beginning Running Code in Section 5")
print("=======================================================================")

def compute_roc(master, model_params):

	dep_var = model_params['dep_var']

	true = master[(master['theta_year'] == master['year'])&
		(master['year'] >= 1996)]\
		[['countryid', 'year', model_params['dep_var']]]

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

	true = true.merge(preds, how = 'left', on = ['countryid', 'year'])

	true = true.dropna(axis = 0)

	overall = roc_curve(true[dep_var], true['predictions'])
	within = roc_curve(true[dep_var], true['within_pred'])

	overall_auc = roc_auc_score(true[dep_var], true['predictions'])
	within_auc = roc_auc_score(true[dep_var], true['within_pred'])
	
	#fig = plt.figure()

	plt.plot(overall[0], overall[1], 'b', color = 'black')
	plt.plot(within[0], within[1], 'b', color = 'black', linestyle = 'dashed')

	plt.text(0.02, 0.07, r'Overall AUC = ' + str(round(overall_auc, 2)), fontsize = 10)
	plt.text(0.02, 0.0, r'Within AUC = ' + str(round(within_auc, 2)), fontsize = 10)

	plt.show()

	# return(true)

print("=======================================================================")
print("Finished Running Code in Section 4")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 5<<<<<<<<<##########################
################################################################################


