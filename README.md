
Author: Chun Hung

We used notebook as the tool to run the data science experiments.

To obtain the cohort dataset for the descriptive analysis / model training, visit
## 1. Data Analysis notebook.
This notebook will load the raw dataset, preprocess it in each steps, and store the cohort dataset, and run the descriptive analysis. The raw dataset is not accessible to public.

## 2. Model Experiment notebook.
This notebook takes in cohort dataset (preprocessed), and run multiple model training, as well as model analysis. To access the latest model we tried, after running the first cell, go to section 4 Go All, run the first cell to read all features, then go to 6 Final. Section 4 Go All is to use all raw features in the model, Section 6 Final is to use features which dropped the insignificant features from the set of all features.

We use extract.py and evaluate.py from utils/, extract.py will preprocess the raw features of each encounter, evaluate.py contains functions to plot the model results.

