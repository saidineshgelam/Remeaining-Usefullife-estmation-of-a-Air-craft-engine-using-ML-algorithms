
import pandas as pd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(324)
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)

# Define the path to the input video file
training_data  = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/flight_data2.txt", sep = "\s+", header = None) # Load the training data
true_rul_ = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/true_rul_1.txt", sep = "\s+", header = None) # Load the true RUL values
testing_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/testing_flight_data.txt", sep = "\s+", header = None) # Load the test data

def process_targets_(lenth_of_data, early_rul_ = None):
    """
    Takes data length and early RUL (Remaining Useful Life) as input and creates target RUL.

    Parameters:
    - lenth_of_data (int): Length of the data.
    - early_rul_ (int, optional): Early RUL value. If None, target RUL is created in reverse order.

    Returns:
    - target_rul (numpy.ndarray): Array of target RUL values.
    """
 
    if early_rul_ == None:   # This condition is needed when early rul is not provided
        return np.arange(lenth_of_data-1, -1, -1) # Target RUL is created in reverse order
    else:  # This condition is needed when early rul is provided
        early_rul_duration_ = lenth_of_data - early_rul_ # Duration of early rul
        if early_rul_duration_ <= 0:    # This condition is needed when early rul is larger than lenth_of_data of an engine
            target_array = np.arange(lenth_of_data-1, -1, -1) # Target RUL is created in reverse order
            return target_array 
        else: # This condition is needed when early rul is smaller than lenth_of_data of an engine
            target_array = np.append(early_rul_*np.ones(shape = (early_rul_duration_,)), np.arange(early_rul_-1, -1, -1)) # Target RUL is created in reverse order
            return target_array 


def create_batches_input_and_targets(input_data_, target_data_ = None, batch_window_length  = 1, shift_among_batches = 1): # This function is used to create batches of input data and target data
    """
    Processes input data and creates input data batches with corresponding target data.

    Parameters:
    - input_data_ (numpy.ndarray): Input data.
    - target_data_ (numpy.ndarray, optional): Target data. If None, only input data batches are created.
    - batch_window_length (int, optional): Length of each input data batch_set.
    - shift_among_batches (int, optional): Shift value for creating input data batches.

    Returns:
    - output_data_batches (numpy.ndarray): Processed input data batches.
    - output_targets_ (numpy.ndarray, optional): Processed target data.
    """
    
    num_of_batches_ = int(np.floor((len(input_data_) - batch_window_length )/shift_among_batches)) + 1 # Number of batches
    num_of_features = input_data_.shape[1] # Number of features
    output_data_batches = np.repeat(np.nan, repeats = num_of_batches_ * batch_window_length  * num_of_features).reshape(num_of_batches_, batch_window_length ,
                                                                                                  num_of_features) # Output data batches
    if target_data_ is None:  # This condition is needed when target data is not provided
        for batch_set  in range(num_of_batches_): # Loop over number of batches
            output_data_batches[batch_set ,:,:] = input_data_[(0+shift_among_batches*batch_set ):(0+shift_among_batches*batch_set +batch_window_length ),:] # Assign input data to output data batches
        return output_data_batches # Return output data batches
    else: # This condition is needed when target data is provided
        output_targets_ = np.repeat(np.nan, repeats = num_of_batches_)  # Output targets
        for batch_set  in range(num_of_batches_): # Loop over number of batches 
            output_data_batches[batch_set ,:,:] = input_data_[(0+shift_among_batches*batch_set ):(0+shift_among_batches*batch_set +batch_window_length ),:] # Assign input data to output data batches
            output_targets_[batch_set ] = target_data_[(shift_among_batches*batch_set  + (batch_window_length -1))] # Assign target data to output targets
    
        return output_data_batches, output_targets_


def processing_test_data(test_data_for_an_engine, batch_window_length , shift_among_batches, num_test_windows = 1): # This function is used to process test data
    """
    Processes test data for a specific engine and creates test data batches.

    Parameters:
    - test_data_for_current_engine (numpy.ndarray): Test data for a specific engine.
    - batch_window_length (int): Length of each test data batch_set.
    - shift_among_batches (int): Shift value for creating test data batches.
    - num_test_windows (int, optional): Number of test data windows to take for each engine.

    Returns:
    - batched_test_data_for_an_engine_ (numpy.ndarray): Processed test data batches.
    - num_test_windows (int): Number of test data windows used.
    """
    max_num_test_batches_ = int(np.floor((len(test_data_for_an_engine) - batch_window_length )/shift_among_batches)) + 1 # Maximum number of test batches 
    if max_num_test_batches_ < num_test_windows: # This condition is needed when maximum number of test batches is smaller than num_test_windows 
        required_length = (max_num_test_batches_ -1)* shift_among_batches + batch_window_length 
        batched_test_data_for_an_engine_ = create_batches_input_and_targets(test_data_for_an_engine[-required_length:, :],
                                                                          target_data_ = None,
                                                                          batch_window_length  = batch_window_length , shift_among_batches = shift_among_batches) # Create batches of test data 
        extracted_num_test_windows = max_num_test_batches_ # Extracted number of test windows
        return batched_test_data_for_an_engine_, extracted_num_test_windows # Return batched test data and extracted number of test windows
    else:
        required_length = (num_test_windows - 1) * shift_among_batches + batch_window_length  # Required length
        batched_test_data_for_an_engine_ = create_batches_input_and_targets(test_data_for_an_engine[-required_length:, :], 
                                                                          target_data_ = None,
                                                                          batch_window_length  = batch_window_length , shift_among_batches = shift_among_batches) # Create batches of test data
        extracted_num_test_windows = num_test_windows # Extracted number of test windows
        return batched_test_data_for_an_engine_, extracted_num_test_windows # Return batched test data and extracted number of test windows



batch_window_length  = 1 # Batch window length
shift_among_batches = 1 # Shift among batches
early_rul_ = None # Early RUL
processed_train_data_ = [] # Processed train data
processed_train_targets_ = [] # Processed train targets


num_test_windows = 20 # Number of test windows
processed_test_data = [] # Processed test data
num_test_windows_list = [] # Number of test windows list

columns_to_be_eliminated = [0,1,4] # Columns to be eliminated
 
num_machines_ = np.min([len(training_data [0].unique()), len(testing_data[0].unique())]) # Number of machines

for i in np.arange(1, num_machines_ + 1): # Loop over number of machines
 
    temp_train_data = training_data [training_data [0] == i].drop(columns=columns_to_be_eliminated).values # Temporary train data
    temp_test_data = testing_data[testing_data[0] == i].drop(columns=columns_to_be_eliminated).values # Temporary test data
 
    # Verify if data of given window length can be extracted from both training and test data
    if (len(temp_test_data) < batch_window_length ): # This condition is needed when length of test data is smaller than batch window length
        print("Test engine {} doesn't have enough data for batch_window_length  of {}".format(i, batch_window_length )) # Print message
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")
    elif (len(temp_train_data) < batch_window_length ): # This condition is needed when length of train data is smaller than batch window length
        print("Train engine {} doesn't have enough data for batch_window_length  of {}".format(i, batch_window_length )) # Print message
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.")

    temp_train_targets = process_targets_(lenth_of_data = temp_train_data.shape[0], early_rul_ = early_rul_) # Temporary train targets
    data_for_a_machine, targets_for_a_machine = create_batches_input_and_targets(temp_train_data, temp_train_targets,
                                                                                batch_window_length  = batch_window_length , shift_among_batches = shift_among_batches) # Create batches of input data and target data

    # Prepare test data
    test_data_for_an_engine, num_windows = processing_test_data(temp_test_data, batch_window_length  = batch_window_length , shift_among_batches = shift_among_batches,
                                                             num_test_windows = num_test_windows) # Process test data

    processed_train_data_.append(data_for_a_machine) # Append data for a machine
    processed_train_targets_.append(targets_for_a_machine) # Append targets for a machine

    processed_test_data.append(test_data_for_an_engine) # Append test data for an engine
    num_test_windows_list.append(num_windows) # Append number of windows

processed_train_data_ = np.concatenate(processed_train_data_) # Concatenate processed train data
processed_train_targets_ = np.concatenate(processed_train_targets_)     # Concatenate processed train targets
processed_test_data = np.concatenate(processed_test_data) # Concatenate processed test data
true_rul_ = true_rul_[0].values # True RUL values

# Shuffle data
index = np.random.permutation(len(processed_train_targets_))
processed_train_data_, processed_train_targets_ = processed_train_data_[index], processed_train_targets_[index] # Shuffle train data and targets



processed_train_data_ = processed_train_data_.reshape(-1, processed_train_data_.shape[2]) # Reshape train data
processed_test_data = processed_test_data.reshape(-1, processed_test_data.shape[2]) # Reshape test data


dtrain = xgb.DMatrix(processed_train_data_, label = processed_train_targets_) # Create DMatrix for train data
dtest = xgb.DMatrix(processed_test_data) # Create DMatrix for test data

num_rounds = 300 # Number of rounds
params = {"max_depth":3, "learning_rate":1, "objective":"reg:squarederror"} # Parameters
bst = xgb.train(params, dtrain, num_boost_round = num_rounds, evals = [(dtrain, "Train")], verbose_eval = 50) # Train the model

rul_pred = bst.predict(dtest) # Predict RUL values

# First split predictions according to number of windows of each engine 
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1]) # Split predictions
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))  # Calculate mean predictions
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)] # Loop over predictions and number of windows
RMSE = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine)) # Calculate RMSE
print("RMSE: ", RMSE) # Print RMSE

plt.plot(true_rul_, label = "True RUL", color = "red")
plt.plot(mean_pred_for_each_engine, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()
# print(preds_for_last_example)

param_grid = [(max_depth, eta) for max_depth in np.arange(2,6) for eta in np.array([0.001, 0.01, 0.1, 0.3, 1])] # Parameter grid

min_rmse = np.inf # Minimum RMSE
best_params = None # Best parameters
params = dict() # Parameters
params["objective"] = "reg:squarederror" # Objective
for max_depth, eta in param_grid: # Loop over parameter grid 
    print("max_depth: {}, eta: {}".format(max_depth, eta)) # Print max_depth and eta

    params["max_depth"] = max_depth # Max depth
    params["eta"] = eta # Eta

    cv_res = xgb.cv(params, dtrain, num_boost_round= num_rounds, early_stopping_rounds= 10, nfold = 10, seed = 789) # Cross validation

    best_rmse_val = cv_res["test-rmse-mean"].min() # Best RMSE value
    best_num_rounds = cv_res["test-rmse-mean"].argmin() + 1 # Best number of rounds

    print("RMSE: {} in {} rounds".format(best_rmse_val, best_num_rounds)) # Print RMSE and number of rounds
    print()

    if best_rmse_val < min_rmse: # This condition is needed when best RMSE value is smaller than minimum RMSE
        min_rmse = best_rmse_val # Minimum RMSE
        best_params = (max_depth, eta, best_num_rounds) # Best parameters

print("Best parameters are: Max_depth= {}, eta= {}, num_rounds = {}. Corresponding RMSE: {}".format(best_params[0],
                                                                                                    best_params[1],
                                                                                                    best_params[2],
                                                                                                    min_rmse)) # Print best parameters

params_tuned = {"max_depth":5, "eta":0.1, "objective":"reg:squarederror"} # Tuned parameters
bst_tuned = xgb.train(params_tuned, dtrain, num_boost_round= 70) # Train the model

rul_pred_tuned = bst_tuned.predict(dtest) # Predict RUL values

preds_for_each_engine_tuned = np.split(rul_pred_tuned, np.cumsum(num_test_windows_list)[:-1]) # Split predictions 
mean_pred_for_each_engine_tuned = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows)) # Calculate mean predictions
                                   for ruls_for_each_engine, num_windows in zip(preds_for_each_engine_tuned,
                                                                                num_test_windows_list)] # Loop over predictions and number of windows
RMSE_tuned = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine_tuned)) # Calculate RMSE
print("RMSE after hyperparameter tuning: ", RMSE_tuned) # Print RMSE

params_tuned = {"max_depth":5, "eta":0.01, "objective":"reg:squarederror"} # Tuned parameters
bst_tuned = xgb.train(params_tuned, dtrain, num_boost_round= 150) # Train the model

rul_pred_tuned = bst_tuned.predict(dtest) # Predict RUL values

preds_for_each_engine_tuned = np.split(rul_pred_tuned, np.cumsum(num_test_windows_list)[:-1]) # Split predictions
mean_pred_for_each_engine_tuned = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows)) # Calculate mean predictions
                                   for ruls_for_each_engine, num_windows in zip(preds_for_each_engine_tuned,
                                                                                num_test_windows_list)] # Loop over predictions and number of windows
RMSE_tuned = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine_tuned)) # Calculate RMSE
print("RMSE after hyperparameter tuning: ", RMSE_tuned) # Print RMSE

indices_of_last_examples = np.cumsum(num_test_windows_list) - 1 # Indices of last examples



preds_for_last_example = np.concatenate(preds_for_each_engine_tuned)[indices_of_last_examples] # Predictions for last example

RMSE_new = np.sqrt(mean_squared_error(true_rul_, preds_for_last_example)) # Calculate RMSE
print("RMSE (Taking only last examples): ", RMSE_new) # Print RMSE
# Plot true and predicted RUL values

plt.plot(true_rul_, label = "True RUL", color = "red")
plt.plot(preds_for_last_example, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()
print(preds_for_last_example)


xgb.plot_importance(bst_tuned)