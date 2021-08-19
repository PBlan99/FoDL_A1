# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

# note that this is just to use the train-test splitting method, all the perceptron (and improvement) algorithm code is done from scratch 
from sklearn.model_selection import train_test_split

# NOTE: let's have everything as a row vector, the w and the x  

# load in data into a pandas dataframe

df = pd.read_table("diabetes_scale.txt", sep='\s+', header=None)
df.fillna(0, inplace=True)  # there are 9 samples which have a missing last column in the diabetes_scale data set, the missing feature value is assigned a zero
df = df[df[8] != 0]  # these 9 samples where the final column is colummn are removed from the data

df.reset_index(drop=True, inplace=True) # reset the numerical indices, since some were removed with the 9 samples

# use this selection to count how many samples belong to class 1 (the class is stored in the first column), so that class weight correction factors can be calculated
df_select_for_count = df[df[0] == 1]

# remove the n: at the beginning of each string value in dataframe
df[1] = df[1].str.slice_replace(0, 2, '')
df[2] = df[2].str.slice_replace(0, 2, '')
df[3] = df[3].str.slice_replace(0, 2, '')
df[4] = df[4].str.slice_replace(0, 2, '')
df[5] = df[5].str.slice_replace(0, 2, '')
df[6] = df[6].str.slice_replace(0, 2, '')
df[7] = df[7].str.slice_replace(0, 2, '')
df[8] = df[8].str.slice_replace(0, 2, '')

# now convert all the strings to floats
df[1] = pd.to_numeric(df[1], downcast="float")
df[2] = pd.to_numeric(df[2], downcast="float")
df[3] = pd.to_numeric(df[3], downcast="float")
df[4] = pd.to_numeric(df[4], downcast="float")
df[5] = pd.to_numeric(df[5], downcast="float")
df[6] = pd.to_numeric(df[6], downcast="float")
df[7] = pd.to_numeric(df[7], downcast="float")
df[8] = pd.to_numeric(df[8], downcast="float")

# split into targets and features
df_targets = df[0]
df_features = df.drop(0,axis=1)
# need to add the unity dummy feature which is to be multiplied by the bias weight
df_features[9] = 1.0

# split data into training and test set with random shuffle and maintaining the proportions of the classes
train_features, test_features, train_targets, test_targets = train_test_split(df_features, df_targets, test_size=0.25, random_state=42, stratify=df_targets)



# need to initialise the 9 weights randomly, with values between 0 and 1.

# function to test if the predicted value is wrong, if it is add to misclassification counter and return a zero, if it is correct then return the multiplication of the scalar yi, the row vector xi, and the class weight correction factor (496/759) for -1 class which is less frequent, and the class weight correction factor (263/759) for +1 class which is more frequent 

# will make plots of percentage that is misclassified, as a function of iterations, for different learning rates

# then you will have adaline as an improvement

# then maybe mini-batch gradient descent for both of these methods

# and also time the runtime of the method