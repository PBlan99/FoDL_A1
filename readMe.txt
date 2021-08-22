To run the code from a command terminal, simply write:

>> python perceptron.py

or

>> python adaline.py

The iteration number will be displayed on the screen (goes up to 3000 iterations), and at the end the training and test errors will be displayed on the screen. A figure showing training error vs iterations is produced by the code and saved in the working directory.

Note that the only algorithmic difference between the perceptron and adaline scripts is in the way the weight corrections are calculated, in the function: check_prediction_and_return_weight_correction_for_one_sample(yi, xi_vector, w_vector)



