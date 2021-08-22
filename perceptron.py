# -*- coding: utf-8 -*-
"""
Author: Pablo Nicolas Blanco
Student ID: a1609603

Perceptron Algorithm for FoDL Assignment 1
"""

import pandas as pd
import numpy as np
from random import seed
from random import random
import matplotlib.pyplot as plt

# note that this is just to use the train-test splitting method, all the perceptron (and improvement) algorithm code is done from scratch 
from sklearn.model_selection import train_test_split

# DATA PREPROCESSING

print("Data preprocessing...")

# load in data into a pandas dataframe

df = pd.read_table("diabetes_scale.txt", sep='\s+', header=None)
df.fillna(0, inplace=True)  # there are 9 samples which have a missing last column in the diabetes_scale data set, the missing feature value is assigned a zero
df = df[df[8] != 0]  # these 9 samples where the final column is missing (now a zero) are removed from the data

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
targets_df = df[0]
features_df = df.drop(0,axis=1)
# need to add the unity dummy feature which is to be multiplied by the bias weight
features_df[9] = 1.0

# split data into training and test set with random shuffle and maintaining the proportions of the classes
train_features_df, test_features_df, train_targets_df, test_targets_df = train_test_split(features_df, targets_df, test_size=0.25, random_state=42, stratify=targets_df)

# convert dataframes to numpy arrays
train_features_matrix = train_features_df.to_numpy()
test_features_matrix = test_features_df.to_numpy()
train_targets_vector = train_targets_df.to_numpy()
test_targets_vector = test_targets_df.to_numpy()

# TRAINING OF PERCEPTRON

# function to test if the predicted value is wrong, if it is then return a boolean flag to add to the misclassification count for the iteration and also return the weight correction which is the multiplication of the scalar yi, the row vector xi, and a class frequency correction factor (496/759) for -1 class which is less frequent or (263/759) for +1 class which is more frequent 
# if prediction is right, the weight correction is zero
def check_prediction_and_return_weight_correction_for_one_sample(yi, xi_vector, w_vector):
    # results list stores a misclassification boolean flag and the weight correction factor
    results = [False, 0*xi_vector]
    
    if (yi*np.dot(xi_vector,w_vector) < 0):
        results[0] = True
        if (yi == 1):
            results[1] = (263/759)*yi*xi_vector  # class frequency correction factor (263/759) for +1 class which is more frequent
        else:
            results[1] = (496/759)*yi*xi_vector # class frequency correction factor (496/759) for -1 class which is less frequent
    
    return results

    
# seed the pseudo random number generator 
seed(1)
# initialise the 9 weights using pseudo random numbers, with values between 0 and 1.
w_vector = np.array([random(),random(),random(),random(),random(),random(),random(),random(),random()])

# set learning rate
eta = 0.01

# at each iteration of the algorithm, iterate through all the samples and call the function for every sample, 
num_iterations = 3000 # (including the zeroth iteration)
list_iteration_numbers = []
list_misclassification_counts_at_each_iteration = []

for i in range(num_iterations):
    
    print("The iteration is: " + str(i))
    list_iteration_numbers.append(i)
    
    misclassification_count = 0
    list_weight_corrections_for_each_sample = []
    
    # iterate through all the samples by iterating through the rows in the features matrix (each row is a sample vector xi) and the rows in the targets vector (each row is a scalar class target yi), using the same index i since the numpy arrays have the same indices, starting from zero
    for i in range(train_targets_vector.size):
        xi_vector = train_features_matrix[i] # this obtains a row from the features matrix, a row is a single sample
        yi = train_targets_vector[i]
        results_list = check_prediction_and_return_weight_correction_for_one_sample(yi, xi_vector, w_vector)
        # if the prediction is wrong, add to misclassificatoin count
        if results_list[0]:
            misclassification_count += 1
        # add the weight correction to the list for each sample
        list_weight_corrections_for_each_sample.append(results_list[1])
            
    # store misclassification count for this iteration
    list_misclassification_counts_at_each_iteration.append(misclassification_count)
    
    # update weights in w_vector using the learning rate and the average of the weight corrections for all the samples
    w_vector = w_vector + eta*sum(list_weight_corrections_for_each_sample)/len(list_weight_corrections_for_each_sample)
        

training_percentage_error_at_each_iteration = (100/train_targets_vector.size) * np.array(list_misclassification_counts_at_each_iteration) 
iteration_numbers = np.array(list_iteration_numbers)
plt.figure()
plt.plot(iteration_numbers,training_percentage_error_at_each_iteration)
plt.title("Perceptron training error vs iterations, learning rate=0.01")
plt.xlabel("Iteration")
plt.ylabel("Training error (%)")
plt.ylim(0,85)
plt.xlim(list_iteration_numbers[0],list_iteration_numbers[-1]+1)
plt.grid()
plt.savefig("perceptron_training_error_vs_iterations.png", format='png')

print("The perceptron training error is: " + str(round(100*misclassification_count/train_targets_vector.size,1)) + " %")

# TESTING OF PERCEPTRON

# count the number of misclassifications when the perceptron model with the trained weights is applied to the test daata 
test_misclassification_count = 0

# iterate through all the samples by iterating through the rows in the features matrix (each row is a sample vector xi) and the rows in the targets vector (each row is a scalar class target yi), using the same index i since the numpy arrays have the same indices, starting from zero
for i in range(test_targets_vector.size):
    test_xi_vector = test_features_matrix[i] # this obtains a row from the features matrix, a row is a single sample
    test_yi = test_targets_vector[i]
    test_results_list = check_prediction_and_return_weight_correction_for_one_sample(test_yi, test_xi_vector, w_vector)
    # if the prediction is wrong, add to misclassificatoin count
    if test_results_list[0]:
        test_misclassification_count += 1
        
print("The perceptron test error is: " + str(round(100*test_misclassification_count/test_targets_vector.size,1)) + " %")
