# NumPy
import numpy as np

# Matplotlib
from matplotlib import pyplot as plt

# Pandas
import pandas as pd

# Preprocessing from sklearn
from sklearn import preprocessing

# Linear regression function from sklearn
from sklearn import linear_model as lm

# Metric function from sklearn
from sklearn import metrics

# Variance Inflation Factor function from statsmodel
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Function to split train and test datasets from sklearn
from sklearn.model_selection import train_test_split

# Function for curve fitting from scipy
from scipy.optimize import curve_fit

# Function from sklearn.linear_model
from sklearn.linear_model import LogisticRegression, perceptron

# API from statsmodels
import statsmodels.api as sm

# StandardScaler function from sklearn
from sklearn.preprocessing import StandardScaler

# Tree function from sklearn
from sklearn import tree

# SVM classifier function from sklearn
from sklearn.svm import SVC

# kNN classifier function from sklearn
from sklearn.neighbors import KNeighborsClassifier

# Functions from sklearn.cluster
from sklearn.cluster import KMeans, AgglomerativeClustering

# PCA function from sklearn
from sklearn.decomposition import PCA

# 3D plot function from mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

# Cross validation score function from sklearn
from sklearn.cross_validation import cross_val_score

# Function from sklearn.ensemble
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Functions from sklearn.neural_network
from sklearn.neural_network import MLPClassifier, BernoulliRBM

# Print libraries/functions imported
print(
'''
Libraries/functions imported:
numpy (np)
matplotlib.pyplot (plt)
pandas (pd)
sklearn (preprocessing)
sklearn.linear_model (lm)
sklearn (metrics)
statsmodels.stats.outliers_influence (variance_inflation_factor)
sklearn.model_selection (train_test_split)
scipy.optimize (curve_fit)
sklearn.linear_model (LogisticRegression, perceptron)
statsmodels.api (sm)
sklearn.preprocessing (StandardScaler)
sklearn (tree)
sklearn.svm (SVC)
sklearn.neighbors (KNeighborsClassifier)
sklearn.cluster (KMeans, AgglomerativeClustering)
sklearn.decomposition (PCA)
mpl_toolkits.mplot3d (Axes3D)
sklearn.cross_validation (cross_val_score)
sklearn.ensemble (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier)
sklearn.neural_network (MLPClassifier, BernoulliRBM)

'''
)

# User Defined Function to Calculate Root Mean Squared Error
def RMSE(y_true, y_pred):
	"""
	y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.
	
	y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
	
	Returns float or ndarray of floats
	"""
	return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
	
# Print user defined functions imported
print(
"""
User defined functions:
Root Mean Squared Error (RMSE)
"""
)