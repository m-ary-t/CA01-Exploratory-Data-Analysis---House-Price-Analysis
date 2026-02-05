# CA01-Exploratory-Data-Analysis---House-Price-Analysis
This is an Exploratory Data Analysis (EDA) and data preparation on housing dataset in order to make the training data analytics‑ready for future house price prediction tasks. The goal of the analysis is to understand the structure, quality, and relationships within the data prior to any machine learning modeling.
The work follows the three‑part EDA framework: Data Understanding, Data Pre‑processing, and Collinearity Analysis. All analysis, observations, and conclusions are contained within a single Jupyter notebook

## Data Understanding
The analysis begins by loading the training dataset and examining its structure, variable types, and overall composition. Features are categorized into numerical, nominal categorical, and ordinal categorical variables based on the dataset documentation. Special attention is given to variables where missing values represent the absence of a feature (such as basements, garages, or pools). In these cases, missing entries are recoded explicitly as "None" to preserve their meaning and prevent misinterpretation as data quality issues.
Ordinal categorical variables, such as quality and condition ratings, are converted into ordered categories that reflect their natural ranking. This ensures that each variable’s data type accurately represents its meaning and prepares the data for proper encoding later in the analysis.
An automated Data Quality Report is generated using YData Profiling to summarize distributions, missing values, extreme values, skewness, and correlations. The findings from this report guide all subsequent preprocessing decisions.

## Pre‑Processing
Based on the issues identified during data understanding, several preprocessing steps are applied to improve data quality:
Missing values are addressed. For example, LotFrontage values are imputed using the mean due to their moderate missingness, while missing garage construction years are handled consistently with homes that do not have garages.
Continuous numerical variables such as area measurements and year‑based features are scaled to ensure comparable magnitudes across features. Count variables and identifier‑style features are intentionally excluded from scaling.
Categorical variables are encoded based on their data type. Ordinal variables are encoded using numerical codes that preserve their ranking. Nominal variables are one‑hot encoded with a reference category dropped to avoid multicollinearity introduced by dummy variables.

## Post-Processing
The post-processing section focuses on identifying and addressing collinearity. Correlations are computed across all numeric features, and a heatmap visualization is used for an initial overview. To make the analysis more interpretable, highly correlated feature pairs (absolute correlation ≥ 0.80) are put into a structured table. Recommendations are given, like removing duplicated categorical indicators that convey the same information.

## Required Libraries
The following Python libraries are required to run this notebook end‑to‑end:
- pandas; Used for data loading, cleaning, transformation, type conversion, encoding, and general data manipulation.
- numpy; Used for numerical operations, array handling, and correlation matrix calculations.
- matplotlib; Used for static data visualizations, including plots and correlation heatmaps.
- seaborn; Used to create enhanced statistical visualizations, particularly the correlation heatmap.
- scikit‑learn; Used for Feature scaling (StandardScaler) & One‑hot encoding of nominal categorical variables (OneHotEncoder)
- ydata‑profiling; Used to generate an automated Data Quality Report summarizing missing values, distributions, skewness, imbalance, and correlations.

### How They Were Downloaded
!pip install ydata-profiling -q
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import nbformat as nbf
import seaborn as sns

