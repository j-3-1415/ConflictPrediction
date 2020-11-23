##########################################################################
# This script imports the data and merges the dependent variable and
# covariates with the theta loadings. It returns a master data frame that
# can be used for all subsequent analysis.
##########################################################################


# Import libraries
import os
import pandas as pd

global currDir  # Define global variable of current directory
global local  # Define global variable of input for file source
global FileDict  # Define global variable for file importation

currDir = os.getcwd()  # Define current directory based on python script
# currDir = '/Users/JPichelmann/Dropbox/PanelData2020/mastercode'

# Get user input whether files should be imported locally or from Dropbox
local = input("""Import data Locally? 'True' or 'False':
    If 'False' Then Comes from Dropbox Online: """)

# Create a dictionary of files paths for dropbox. Used in FileRead function
FileDict = {
    'Topics': ["wz1azpy6teayyzr", "topics.csv"],
    'TopicID': ["ofgtf9ltq9eb8vt", "TopicID.csv"],
    'CountryID': ["6je8xdzg63krmw5", "countryids.dta"],
    'EventData': ["b1aahhlke2gd1n6", "eventdata.dta"],
    'CompleteMain': ["srgz7t7k5t3omws", "complete_main_new.dta"],
    'CompleteMain_OG': ["qrud7n5ak4pcawk", "complete_main_new_original.dta"],
    'ArticleCount': ["vbz5dh78myqaa32", "article_count.dta"],
    'Theta_1995': ["xqqb03az8dh10xo"], 'Theta_1996': ["kn0n9kuc9342wyi"],
    'Theta_1997': ["el9ltiga8fnyhnf"], 'Theta_1998': ["kt9zo86nsgug18c"],
    'Theta_1999': ["3e90uysagsh5j6y"], 'Theta_2000': ["yird2kmqyfx4qos"],
    'Theta_2001': ["ogbvafkck6uci1f"], 'Theta_2002': ["dxybjz2nmc2wr1g"],
    'Theta_2003': ["47xseavooz97h3i"], 'Theta_2004': ["uymak66cgwh3cqz"],
    'Theta_2005': ["pkcpx7n5kiiwmab"], 'Theta_2006': ["20w1xwm9g3xoqxv"],
    'Theta_2007': ["xnseiqxoahbfmns"],  'Theta_2008': ["bezx6p9wu0t8bxo"],
    'Theta_2009': ["cmfei42pxlmt3ca"], 'Theta_2010': ["na35rztvy187dqi"],
    'Theta_2011': ["r7x0478sjos6six"], 'Theta_2012': ["5y5on73oxb4fgpf"],
    'Theta_2013': ["of7mo0usbkqj82w"],  'Theta_2014': ["vfe2xplgd2c9jw3"],
    'Theta_2015': ["fqdq5ax5sixy8om"],
    'Theta_Full_1995': ["z59vqlbrprczvnh"],
    'Theta_Full_1996': ["zlhr25gr9tlcycg"],
    'Theta_Full_1997': ["nz8fqk6hh61f3v2"],
    'Theta_Full_1998': ["ej30ikb0b3sh88e"],
    'Theta_Full_1999': ["qqo39u38dqcqsve"],
    'Theta_Full_2000': ["5g1v084txwt17yb"],
    'Theta_Full_2001': ["wyh83n76aizguxr"],
    'Theta_Full_2002': ["11oq0swh40sk93q"],
    'Theta_Full_2003': ["v6x25dai1vbb481"],
    'Theta_Full_2004': ["mp0470zj7gag5de"],
    'Theta_Full_2005': ["ppcngl2qd3c53qq"],
    'Theta_Full_2006': ["q3n2p8j4y0iocka"],
    'Theta_Full_2007': ["u1z5j6d6v69dakh"],
    'Theta_Full_2008': ["zyk9x1ti0tc7ry5"],
    'Theta_Full_2009': ["z4rqncmntabs8iy"],
    'Theta_Full_2010': ["9yja3dqqw3rquis"],
    'Theta_Full_2011': ["4wnvqufctpvl1ms"],
    'Theta_Full_2012': ["p5bfd51fbwya7d4"],
    'Theta_Full_2013': ["j3ik866cvobs4rh"],
    'Theta_Full_2014': ["fr5mo5raw1a6s5a"],
    'Theta_Full_2015': ["3yit1p2sr3i7m0q"]}

# Create a function to read in the files based on codes within dictionaary


def FileRead(KeyName):

        # Split the dictionary key into a list
    KeyList = KeyName.split("_")

    # Condition to put together theta file name if needed
    if KeyList[0] == "Theta":
        FileName = "thetas15_alpha3_beta001_all_both"
        if KeyList[1] == "Full":
            FileName += KeyList[2] + ".dta"
        else:
            FileName += "_collapsed" + KeyList[1] + ".dta"
    else:
        FileName = FileDict[KeyName][1]

    # Condition for whether imported locally or from dropbox
    if local:
        path = currDir + "/dataverse_files/data/" + FileName
    else:
        code = FileDict[KeyName][0]  # Get dropbox code
        path = "https://www.dropbox.com/s/" + code + "/" + FileName + "?dl=1"

    # DTA files are STATA files and need a different import function
    if '.dta' in path:
        return(pd.read_stata(path))
    else:
        return(pd.read_csv(path))


# Create a dictionary with labels for each column
labs = {'year': 'Article Year', 'theta_year': 'Topic Year',
        'countryid': 'CountryID', 'pop': 'Country Population',
        'theta0': 'Industry', 'theta1': 'CivicLife1',
        'theta2': 'Asia', 'theta3': 'Sports',
        'theta4': 'Justice', 'theta5': 'Tourism',
        'theta6': 'Politics', 'theta7': 'Conflict1',
        'theta8': 'Business', 'theta9': 'Economics',
        'theta10': 'InterRelations1', 'theta11': 'InterRelations2',
        'theta12': 'Conflict3', 'theta13': 'CivicLife2',
        'theta14': 'Conflict2', 'bdbest1000': "Civil War",
        'bdbest25': "Armed Conflict", 'const': 'Constant'}

# Two main files needed for merge: thetas and complete data including dep var and controls

# get an overview of available data for controls
complete_file = FileRead("CompleteMain")
for col in complete_file.columns:
    print(col)

# specify range of years you want to use
theta_years = list(range(1995, 2015))

# specify the controls you want to use
own = ['region_o', 'subregion_o', 'discrimshare']

# Include the desired interaction variables to be include in master dataframe
interactions = ['childmortality', 'rgdpl', 'democracy', 'avegoodex']
interactions_names = ['ChildMortality', 'RealGDP',
                      'DemocracyIndex', 'AveGoodIndex']
add_labs = dict(zip(interactions, interactions_names))

labs.update(add_labs)  # include them in labels

own.extend(interactions)

# include interaction terms in labels
labs.update({"theta" + str(i) + "_BY_" + inter:
             labs["theta" + str(i)] + " by " + labs[inter]
             for i in range(0, 15) for inter in interactions})

lag_labs = {key: "Lag of " + val for key, val in labs.items()}

all_labs = {'Labs': labs, 'Lag_Labs': lag_labs}

# list to store the ready to go regression data for each year
master_data = []

# Define index columns to merge on
merge_cols = ['year', 'countryid']
# Define the variables needed to construct the dependent variables
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
    complete_file_sub = complete_file.\
        loc[:, complete_file.columns.isin(filter_col)]
    # recode dependent variables

    # merge thetas with the rest of the data
    merged = pd.merge(complete_file_sub, theta_file_sub,
                      how='left', on=['year', 'countryid'])
    merged['theta_year'] = year
    master_data.append(merged)


master = pd.concat(master_data)
master = master.sort_values(by=['theta_year', 'countryid', 'year'])
master2 = master.groupby('countryid', as_index=False)['pop'].mean()
master2.rename({'pop': 'avpop'}, axis=1, inplace=True)
master = master.merge(master2, on='countryid', how='left')
master = master[master['year'] > 1974]
master = master.reset_index().drop('index', axis=1)
master = master.rename(
    columns={'ste_theta' + str(i): 'theta' + str(i) for i in range(0, 15)})
