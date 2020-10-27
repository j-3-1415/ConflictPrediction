######################=======================================>>>>>>>>>>>>>>>>>>>
# Panel Data 2020 Replication Project
# Luca Poll, Jacob Pichelmann, Andrew Boomer
# Last Modified By: Andrew Boomer 09/10/2020
# Modifications: Writing skeleton of code, organizing file structure
######################=======================================>>>>>>>>>>>>>>>>>>>


################################################################################
#######################>>>>>>>NOTES TO GROUP<<<<<<<<<###########################
################################################################################

# Note to fellow group members! I think we can differentiate tasks through the
# table of contents. The next code To-Do is being handled by Jacob, which is
# indicated in section 2 of the table of contents

''' Comment: In my opinion the paper is structured like the following: 
	
	(## no need to replicate ##)
    0) Topic modelling: Preprocessing, LDA, Gibbs sampler

	(### no need to replicate ###)
    1) Argument for relevance of within variation through apllication of model
       to existing models from literature
		1.1 Rainfall: Miguel & Satyanath (2011)
		1.2 External economic shocks and political constrains: Besley & Presson(2011b)
		1.3 Political institution dummies: Goldstone et al. (2010)
		1.4 ICEWS conflict events: Ward et al. (2013)
		1.5 Keyword counts: Chadefaux (2014)
		1.6 Country fixed effects only

    2) Topic model

    	(### very important to REPLICATE ###)
		2.1 Performance of topic model
		    2.1.1 run fixed effects estimation to obtain fitted values
	        2.1.2 take fitted values (overall and within model) and create
	                  binary variable depending on cutoff value c
	        2.1.3 Compare forecast (binary variable) with realized values and create
	              Graph ROC curves of model performance
		    ## note: important to have different samples and time horizons T in mind
		
		(### no need to replicate ###)
		2.2 Robustness checks
	        2.2.1 Conflict incidence
		    2.2.2 Varying number of topics
		    2.2.3 Topics with other confounders
			  2.2.3.1 Political regime dummies, infant mortality, share of
	                  population descriminated against, dummy if neighboring
	                  countries in conflict
			  2.2.3.2 Conflict escalation: ongoing armed conflict as
	                  predictor for civil war
		    2.2.4 Changing definition of conflict
			  2.2.4.1 all types of conflict (+external wars)
			  2.2.4.2 battle-related deaths in internal wars
			  2.2.4.3 data by Besley & Persson (2011b)
			  2.2.4.4 conflict if >0.08 battle death per 1000 inhabitants
			  2.2.4.5 upper bound estimations PRIO
			  2.2.4.6 UNHCR refugees
			  2.2.4.7 Two years forecast horizon
		    2.2.5 By space and time
		    2.2.6 Comparison to conflict events and keyword counts
		    2.2.7 Neural-network approach
		    2.2.8 Logit without country fixed effects

	(### maybe relevant for later but not now ###)
    3) Analysis why topics provide useful forecasting power on time dimension 
		3.1 Topics harmonized to baseline year 2013
		3.2 LASSO (three parameters of selectivity: 100, 150 & 200 for both y
		3.3 Robustness checks (excluding conflict topics)
'''

################################################################################
######################>>>>>>>END NOTES SECTION<<<<<<<<<#########################
################################################################################


################################################################################
#########################>>>>>>>>MODEL NOTES<<<<<<<<<###########################
################################################################################

# Conflict Types
#	Stata Variable Name: "conflict_type"
#		2 = Armed Conflict = 'bdbest25'
#		3 = Civil Way = 'bdbest1000'

# Incidence vs. Onset
#	Stata Variable Name: "cheat" / "included"
#		1 = onset
#		NULL = incidence
#		Remove 1 or NULL for dependent variable

################################################################################
######################>>>>>>>>>END MODEL NOTES<<<<<<<<<#########################
################################################################################


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

global currDir #Define global variable of current directory
global local #Define global variable of input for file source

currDir = os.getcwd() #Define current directory based on python script

#Get user input whether files should be imported locally or from Dropbox
local = input("""Import data Locally? 'True' or 'False': 
	If 'False' Then Comes from Dropbox Online: """)

#Create text input based on true/false flag from user input
local = ['Dropbox', 'Local'][local == 'True']

#Define general filename for theta files
theta = "thetas15_alpha3_beta001_all_both_collapsed{}.dta"
theta_db = "thetas15_alpha3_beta001_all_both_collapsed{}.dta?dl=1"

#Create a dictionary of files paths, both local and from dropbox
#User will use these paths to download data into python
#First layer is file name alias, second layer is choice between local or Dropbox
files = {
	'Topics' : {
		'Dropbox' : "https://www.dropbox.com/s/wz1azpy6teayyzr/topics.csv?dl=1",
		'Local' : "dataverse_files/data/topics.csv"},
	'TopicID' : {
		'Dropbox' : "https://www.dropbox.com/s/ofgtf9ltq9eb8vt/TopicID.csv?dl=1",
		'Local' : "dataverse_files/data/TopicID.csv"},
	'CountryID' : {
		'Dropbox' : "https://www.dropbox.com/s/6je8xdzg63krmw5/countryids.dta?dl=1",
		'Local' : "dataverse_files/data/countryids.dta"},
	'EventData' : {
		'Dropbox' : "https://www.dropbox.com/s/b1aahhlke2gd1n6/eventdata.dta?dl=1",
		'Local' : "dataverse_files/data/eventdata.dta"},
	'CompleteMain' : {
		'Dropbox' : "https://www.dropbox.com/s/srgz7t7k5t3omws/" \
					"complete_main_new.dta?dl=1",
		'Local' : "dataverse_files/data/complete_main_new.dta"},
	'CompleteMain_OG' : {
		'Dropbox' : "https://www.dropbox.com/s/qrud7n5ak4pcawk/" \
					"complete_main_new_original.dta?dl=1",
		'Local' : "dataverse_files/data/complete_main_new_original.dta"},
	'ArticleCount' : {
		'Dropbox' : "https://www.dropbox.com/s/vbz5dh78myqaa32/article_count.csv?dl=1",
		'Local' : "dataverse_files/data/article_count.csv"},
	'Theta_1995' : {
		'Dropbox' : "https://www.dropbox.com/s/xqqb03az8dh10xo/" + theta_db.format(1995),
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed1995.dta"},
	'Theta_1996' : {
		'Dropbox' : "https://www.dropbox.com/s/kn0n9kuc9342wyi/" \
					"thetas15_alpha3_beta001_all_both_collapsed1996.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed1996.dta"},
	'Theta_1997' : {
		'Dropbox' : "https://www.dropbox.com/s/el9ltiga8fnyhnf/" \
					"thetas15_alpha3_beta001_all_both_collapsed1997.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed1997.dta"},
	'Theta_1998' : {
		'Dropbox' : "https://www.dropbox.com/s/kt9zo86nsgug18c/thetas15_alpha3_beta001_all_both_collapsed1998.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed1998.dta"},
	'Theta_1999' : {
		'Dropbox' : "https://www.dropbox.com/s/3e90uysagsh5j6y/thetas15_alpha3_beta001_all_both_collapsed1999.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed1999.dta"},
	'Theta_2000' : {
		'Dropbox' : "https://www.dropbox.com/s/yird2kmqyfx4qos/thetas15_alpha3_beta001_all_both_collapsed2000.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2000.dta"},
	'Theta_2001' : {
		'Dropbox' : "https://www.dropbox.com/s/ogbvafkck6uci1f/thetas15_alpha3_beta001_all_both_collapsed2001.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2001.dta"},
	'Theta_2002' : {
		'Dropbox' : "https://www.dropbox.com/s/dxybjz2nmc2wr1g/thetas15_alpha3_beta001_all_both_collapsed2002.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2002.dta"},
	'Theta_2003' : {
		'Dropbox' : "https://www.dropbox.com/s/47xseavooz97h3i/thetas15_alpha3_beta001_all_both_collapsed2003.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2003.dta"},
	'Theta_2004' : {
		'Dropbox' : "https://www.dropbox.com/s/uymak66cgwh3cqz/thetas15_alpha3_beta001_all_both_collapsed2004.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2004.dta"},
	'Theta_2005' : {
		'Dropbox' : "https://www.dropbox.com/s/pkcpx7n5kiiwmab/thetas15_alpha3_beta001_all_both_collapsed2005.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2005.dta"},
	'Theta_2006' : {
		'Dropbox' : "https://www.dropbox.com/s/20w1xwm9g3xoqxv/thetas15_alpha3_beta001_all_both_collapsed2006.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2006.dta"},
	'Theta_2007' : {
		'Dropbox' : "https://www.dropbox.com/s/xnseiqxoahbfmns/thetas15_alpha3_beta001_all_both_collapsed2007.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2007.dta"},
	'Theta_2008' : {
		'Dropbox' : "https://www.dropbox.com/s/bezx6p9wu0t8bxo/thetas15_alpha3_beta001_all_both_collapsed2008.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2008.dta"},
	'Theta_2009' : {
		'Dropbox' : "https://www.dropbox.com/s/cmfei42pxlmt3ca/thetas15_alpha3_beta001_all_both_collapsed2009.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2009.dta"},
	'Theta_2010' : {
		'Dropbox' : "https://www.dropbox.com/s/na35rztvy187dqi/thetas15_alpha3_beta001_all_both_collapsed2010.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2010.dta"},
	'Theta_2011' : {
		'Dropbox' : "https://www.dropbox.com/s/r7x0478sjos6six/thetas15_alpha3_beta001_all_both_collapsed2011.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2011.dta"},
	'Theta_2012' : {
		'Dropbox' : "https://www.dropbox.com/s/5y5on73oxb4fgpf/thetas15_alpha3_beta001_all_both_collapsed2012.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2012.dta"},
	'Theta_2013' : {
		'Dropbox' : "https://www.dropbox.com/s/of7mo0usbkqj82w/thetas15_alpha3_beta001_all_both_collapsed2013.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2013.dta"},
	'Theta_2014' : {
		'Dropbox' : "https://www.dropbox.com/s/vfe2xplgd2c9jw3/thetas15_alpha3_beta001_all_both_collapsed2014.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2014.dta"},
	'Theta_2015' : {
		'Dropbox' : "https://www.dropbox.com/s/fqdq5ax5sixy8om/thetas15_alpha3_beta001_all_both_collapsed2015.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both_collapsed2015.dta"},
	'Theta_Full_1995' : {
		'Dropbox' : "https://www.dropbox.com/s/z59vqlbrprczvnh/thetas15_alpha3_beta001_all_both1995.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both1995.dta"},
	'Theta_Full_1996' : {
		'Dropbox' : "https://www.dropbox.com/s/zlhr25gr9tlcycg/thetas15_alpha3_beta001_all_both1996.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both1996.dta"},
	'Theta_Full_1997' : {
		'Dropbox' : "https://www.dropbox.com/s/nz8fqk6hh61f3v2/thetas15_alpha3_beta001_all_both1997.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both1997.dta"},
	'Theta_Full_1998' : {
		'Dropbox' : "https://www.dropbox.com/s/ej30ikb0b3sh88e/thetas15_alpha3_beta001_all_both1998.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both1998.dta"},
	'Theta_Full_1999' : {
		'Dropbox' : "https://www.dropbox.com/s/qqo39u38dqcqsve/thetas15_alpha3_beta001_all_both1999.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both1999.dta"},
	'Theta_Full_2000' : {
		'Dropbox' : "https://www.dropbox.com/s/5g1v084txwt17yb/thetas15_alpha3_beta001_all_both2000.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2000.dta"},
	'Theta_Full_2001' : {
		'Dropbox' : "https://www.dropbox.com/s/wyh83n76aizguxr/thetas15_alpha3_beta001_all_both2001.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2001.dta"},
	'Theta_Full_2002' : {
		'Dropbox' : "https://www.dropbox.com/s/11oq0swh40sk93q/thetas15_alpha3_beta001_all_both2002.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2002.dta"},
	'Theta_Full_2003' : {
		'Dropbox' : "https://www.dropbox.com/s/v6x25dai1vbb481/thetas15_alpha3_beta001_all_both2003.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2003.dta"},
	'Theta_Full_2004' : {
		'Dropbox' : "https://www.dropbox.com/s/mp0470zj7gag5de/thetas15_alpha3_beta001_all_both2004.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2004.dta"},
	'Theta_Full_2005' : {
		'Dropbox' : "https://www.dropbox.com/s/ppcngl2qd3c53qq/thetas15_alpha3_beta001_all_both2005.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2005.dta"},
	'Theta_Full_2006' : {
		'Dropbox' : "https://www.dropbox.com/s/q3n2p8j4y0iocka/thetas15_alpha3_beta001_all_both2006.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2006.dta"},
	'Theta_Full_2007' : {
		'Dropbox' : "https://www.dropbox.com/s/u1z5j6d6v69dakh/thetas15_alpha3_beta001_all_both2007.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2007.dta"},
	'Theta_Full_2008' : {
		'Dropbox' : "https://www.dropbox.com/s/zyk9x1ti0tc7ry5/thetas15_alpha3_beta001_all_both2008.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2008.dta"},
	'Theta_Full_2009' : {
		'Dropbox' : "https://www.dropbox.com/s/z4rqncmntabs8iy/thetas15_alpha3_beta001_all_both2009.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2009.dta"},
	'Theta_Full_2010' : {
		'Dropbox' : "https://www.dropbox.com/s/9yja3dqqw3rquis/thetas15_alpha3_beta001_all_both2010.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2010.dta"},
	'Theta_Full_2011' : {
		'Dropbox' : "https://www.dropbox.com/s/4wnvqufctpvl1ms/thetas15_alpha3_beta001_all_both2011.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2011.dta"},
	'Theta_Full_2012' : {
		'Dropbox' : "https://www.dropbox.com/s/p5bfd51fbwya7d4/thetas15_alpha3_beta001_all_both2012.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2012.dta"},
	'Theta_Full_2013' : {
		'Dropbox' : "https://www.dropbox.com/s/j3ik866cvobs4rh/thetas15_alpha3_beta001_all_both2013.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2013.dta"},
	'Theta_Full_2014' : {
		'Dropbox' : "https://www.dropbox.com/s/fr5mo5raw1a6s5a/thetas15_alpha3_beta001_all_both2014.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2014.dta"},
	'Theta_Full_2015' : {
		'Dropbox' : "https://www.dropbox.com/s/3yit1p2sr3i7m0q/thetas15_alpha3_beta001_all_both2015.dta?dl=1",
		'Local' : "dataverse_files/data/thetas15_alpha3_beta001_all_both2015.dta"}
}

#Create a function to read in the files based on paths within dictionary
#All the function needs is the alias of the filename specified in the dict
def FileRead(filename):

	if local == "Local":
		filename = currDir + "/" + files[filename][local]
	else:
		filename = files[filename][local]

	if '.dta' in filename:
		return(pd.read_stata(filename))
	else:
		return(pd.read_csv(filename))

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
	'bdbest25' : "Armed Conflict"}

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
own = ['refugees', 'goodinst']

# in the paper they specify different sets of controls
control_sets = {
	'chadefaux': ['cha_count', 'total_count', 'total_count_count', 'polity2', 'sincelast', 'sincelast_sq', 'sincelast_cu'],
	'chadefauxshort': ['cha_count', 'total_count', 'total_count_count'],
	'ajps': ['democ_3', 'democ_4', 'democ_5', 'democ_6', 'lnchildmortality', 'armedconf4', 'discrimshare'],
	'lasso': ['democ_3', 'democ_4', 'democ_6', 'childmortality', 'discrimshare', 'excludedshare', 'dissgov', 'ethnicgov'],
	'ward': ['high', 'low', 'autoc', 'democ', 'excludedshare', 'excludedshare_sq', 'lnchildmortality',  'armedconf4', 'discrimshare'],
	'events': ['govopp', 'dissgov', 'domgov', 'ethnicgov'],
	'pillars': ['nat_dum', 'nat_dum_goodinst',  'scmem', 'scmem_goodinst',  'scmem_cold', 'scmem_cold_goodinst']
}

# get user input which set of controls to use
print("""Available sets of controls:
- chadefaux: cha_count total_count total_count_count polity2 sincelast sincelast_sq sincelast_cu
- chadefauxshort: cha_count total_count total_count_count
- ajps: democ_3 democ_4 democ_5 democ_6 lnchildmortality  armedconf4 discrimshare
- lasso: democ_3 democ_4 democ_6 childmortality discrimshare excludedshare dissgov ethnicgov
- ward: high low autoc democ excludedshare excludedshare_sq lnchildmortality  armedconf4 discrimshare
- wardzwei: highdum lowdum autoc democ excludedshare excludedshare_sq lnchildmortality  armedconf4 discrimshare
- events: govopp dissgov domgov ethnicgov
- pillars: nat_dum nat_dum_goodinst  scmem scmem_goodinst  scmem_cold scmem_cold_goodinst
""")
controls_choice = input("Please choose 'own' if you specified your own controls or choose instead one of the sets above: ").lower()

if controls_choice != 'own':
	controls = control_sets[controls_choice]
else:
	controls = own

# list to store the ready to go regression data for each year
master_data = []

#Define index columns to merge on
merge_cols = ['year', 'countryid']
#Define the variables needed to construct the dependent variables
covars = ['bdbest25', 'bdbest1000', 'pop']
covars.extend(controls)

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
		df.loc[(labs[col], 'between'), 'Min'] = np.min(data.groupby(time, as_index = False)[col].mean()[col])
		df.loc[(labs[col], 'between'), 'Max'] = np.max(data.groupby(time, as_index = False)[col].mean()[col])
		df.loc[(labs[col], "between"), 'Observations'] = len(data[~np.isnan(data[col])][time].unique())
		df.loc[(labs[col], 'between'), 'Std. Dev.'] = np.sqrt(np.sum(np.power(data.groupby(time)[col].mean() - np.mean(data[col]), 2)) / (df.loc[(labs[col], 'between'), 'Observations'] - 1))

		df.loc[(labs[col], 'within'), 'Mean'] = np.nan
		df.loc[(labs[col], 'within'), 'Min'] = np.min(data[col] + np.mean(data[col]) - data.merge(data.groupby(time, as_index = False)[col].mean().rename({col : 'mean'}, axis = 1), on = [time], how = 'left')['mean'])
		df.loc[(labs[col], 'within'), 'Max'] = np.max(data[col] + np.mean(data[col]) - data.merge(data.groupby(time, as_index = False)[col].mean().rename({col : 'mean'}, axis = 1), on = [time], how = 'left')['mean'])
		df.loc[(labs[col], "within"), 'Observations'] = len(data[~np.isnan(data[col])][indiv].unique())
		df.loc[(labs[col], 'within'), 'Std. Dev.'] =  np.sqrt(np.sum(np.power(data[col] - data.merge(data.groupby(time, as_index = False)[col].mean().rename({col : 'mean'}, axis = 1), on = time, how = 'left')['mean'], 2)) / (data[col].count() - 1))

		df.fillna("", inplace = True)

		return(df)

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

def data_compare(data, test, fit_year, compare_vars, dep_var, onset = True):

	data = data.copy()
	test = test.copy()
	thetas = [col for col in data.columns if ('ste_theta' in col)&(col != 'ste_theta0')]
	
	compare_vars = thetas + compare_vars

	test = test[(test['year'] > 1974)&(test['year'] <= fit_year)&
		(test['avpop'] >= 1000)&(test['tokens'] > 0)&(~test['avpop'].isnull())&
		(~test['tokens'].isnull())&(test['samp'] == 1)]

	if onset:
		test = test[test[dep_var] != 1]
	else:
		test = test[~test[dep_var].isnull()]

	test = test[['countryid', 'year'] + compare_vars]
	test = test.reset_index()
	data = data.reset_index()

	print('The number of rows of the data from STATA: ' + str(test.shape[0]))
	print('The number of rows of the data from Python: ' + str(data.shape[0]))

	for var in compare_vars:

		equality = all(data[var].values == test[var].values)
		print('For the variable ' + str(var) + 
			', the columns in STATA and Python are equal: ' + str(equality))
		if not equality:
			print(data[data[var] != test[var]][['countryid', 'year', var]].head())
			print(test[data[var] != test[var]][['countryid', 'year', var]].head())

def run_model(data, fit_year, dep_var, onset = True, all_indiv = True):

	#Get list of countries that are excluded for each run
	#Make a regression summary text file with each run

	data = data.copy(deep = True)
	data = data[data['theta_year'] == (fit_year + 1)]
	thetas = [col for col in data.columns if ('ste_theta' in col)&(col != 'ste_theta0')]

	data['one_before'] = data.groupby('countryid')[dep_var].shift(-1)
	if onset:
		data['one_before'] = np.where((data['one_before'] == 1)&(data[dep_var] == 0), 1, 0)
	else:
		data['one_before'] = np.where((data['one_before'] == 1)&(data[dep_var] == 1), 1, 0)

	data = data[(data['avpop'] >= 1000)&(~data['avpop'].isnull())]

	data[thetas] = data.groupby('countryid')[thetas].ffill()

	data = data.drop([col for col in data.columns if "_Lag" in col], axis = 1)

	data['tokens'] = data.groupby('countryid')['tokens'].ffill()

	if onset:
		data = data[data[dep_var] != 1]
	else:
		data = data[~data[dep_var].isnull()]

	data = data[(data['tokens'] > 0)&(~data['tokens'].isnull())]

	if not all_indiv:
		total = data.groupby('countryid', as_index = False)[dep_var].sum()
		total = total.rename({dep_var : 'total_conflict'}, axis = 1)
		data = data.merge(total, how = 'left', on = 'countryid')
		data = data[data['total_conflict'] > 0]

	data = data[data['year'] <= fit_year]
	data.set_index(['countryid', 'year'], inplace = True)

	exog = data[thetas]
	exog = sm.add_constant(exog)

	model = PanelOLS(data['one_before'], exog,
		entity_effects = True)
	return(model, data)

def pred_model(data, fit_year, model, include_fixef = False):

	data = data.copy()
	data = data[(data['year'] == (fit_year + 1))&(data['theta_year'] == (fit_year + 1))]
	data.set_index(['countryid', 'year'], inplace = True)
	thetas = [col for col in data.columns if ('ste_theta' in col)&(col != 'ste_theta0')]
	exog = data[thetas]
	exog = sm.add_constant(exog)
	preds = model.fit().predict(exog)
	preds.reset_index(inplace = True)
	preds.rename({'predictions' : 'within_pred'}, axis = 1, inplace = True)

	if include_fixef:
		fixef = model.fit().estimated_effects
		fixef = fixef.reset_index()
		fixef = fixef.groupby('countryid', 
			as_index = False)['estimated_effects'].mean()
		fixef.rename({'estimated_effects' : "FE"}, inplace = True, axis = 1)
		preds = preds.merge(fixef, how = 'left', on = 'countryid')
		preds['overall_pred'] = preds['within_pred'] + np.where(preds['FE'].isnull(), 0, preds['FE'])
		preds.drop('FE', axis = 1, inplace = True)

	return(preds)

compare_vars = ['tokens', 'bdbest25', 'one_before', 'avpop']

# test = pd.read_stata(os.getcwd() + '/dataverse_files/data/test.dta')

model, df = run_model(master, 1995, 'bdbest25', onset = True, all_indiv = True)
print(model.fit())

# data_compare(df, test, 1995, compare_vars, 'bdbest25', onset = True)

preds = pred_model(master, 1995, model, include_fixef = True)
print(preds)

print("=======================================================================")
print("Finished Running Code in Section 4")
print("=======================================================================")

################################################################################
#########################>>>>>>>END SECTION 4<<<<<<<<<##########################
################################################################################


