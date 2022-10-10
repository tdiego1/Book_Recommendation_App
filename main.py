# Data processing and Calculations
import numpy as np
import pandas as pd
import scipy.stats

#Visualizations
import seaborn as sns

#Similarity
from sklearn.metrics.pairwise import cosine_similarity

#Read in data
ratings = pd.read_csv('data/Ratings.csv')
ratings.head()


