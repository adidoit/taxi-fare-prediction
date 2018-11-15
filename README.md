## Requirements
Use requirements.txt if needed for version

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import feather
import re
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

## Data 

Please download data from kaggle 
https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data

