# Conflict Prediction

This project was done in collaboration with Jacob Pichelmann and Luca Poll.

## Overview
This paper was the final project for a Panel Data class during my masters program at the Toulouse School of Economics. It was a replication and extension of the paper "Reading between the lines: Prediction of political violence using newspaper text" which can be found [here](https://www.repository.cam.ac.uk/bitstream/handle/1810/302412/mueller%20and%20rauh%202018.pdf?sequence=1). The paper investigates the limitations to conflict prediction with country specific fixed effects, such as geography, climate, and ethnic fractionalization. The authors introduce additional relevant data with sentiment analysis of newspaper text through topic clustering. We use the Blundell-Bond panel model via GMM estimation. A major technical contribution of this project was building the blundell-bond panel model from scratch in Python. While it is available out of the box in STATA, it was not available at the time of this project in Python. Main results include the Blundell-Bond coefficients and ROC curves of the prediction accuracy.

## Data Sources
We use the data provided by the authors, which includes the panel of topic clusters, levels of conflict, and other exogenous variables related to conflict prediction.

## Tools
The paper was built in Latex. The coding was done in Python, using the Pandas, Numpy, Sklearn, LinearModels, Plotly, Geopandas, Statsmodels, and Tkinter packages.

<!-- ## Getting Started

1. Unpack the zip folder in the directory you want to store the code in.
2. The code can be executed in two ways: 
    1. Use your editor of choice and run the script Main.py.
    2. Run Main.py straight from the terminal by first navigating to the code folder
    using 'cd yourchoiceofdirectory/mastercode/Code' and then executing the code by
    typing either 'python3 Main.py' or 'python Main.py' depending on your executable. 
    
    ![terminalrun](run_main.png)

### Prerequisites

* Python 3.8 needs to be installed on your machine in order to be able to execute the code.

* All additional libraries used will be installed automatically when executing the program.

## Code Execution

Executing Main.py prompts two selection windows. 
1. Choose what part of the code you want to run, either the regressions and ROC curves or
the descriptive plots. 
![selectpart](select_part.png)
2. Choose the source of the data download (for ease of use please selection option 'Download From Dropbox)
![selectdata](select_data.png)

This is all that has to be done. The console output will inform you about the progress and all results
will be stored in the 'Report' folder. -->
