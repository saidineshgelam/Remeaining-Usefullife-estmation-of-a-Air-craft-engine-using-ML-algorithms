
import sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(324)
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler



np.random.seed(44)
pd.set_option('display.max_columns', None)
training_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/flight_data2.txt", sep = "\s+", header = None) # Load the training data
testing_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/testing_flight_data.txt", sep = "\s+", header = None) # Load the testing data
true_rul_ = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/true_rul_1.txt", sep = "\s+", header = None) # Load the true RUL values

def process_targets(lenth_of_data, early_rul_ = None): # Function to create target RUL values
    """
    Takes data length and early RUL (Remaining Useful Life) as input and creates target RUL.

    Parameters:
    - lenth_of_data (int): Length of the data.
    - early_rul_ (int, optional): Early RUL value. If None, target RUL is created in reverse order.

    Returns:
    - target_rul (numpy.ndarray): Array of target RUL values.
    """ 
    if early_rul_ == None: # If early RUL is not provided, create target RUL in reverse order
        return np.arange(lenth_of_data-1, -1, -1) # Create target RUL in reverse order
    else: # If early RUL is provided, create target RUL with early RUL and then in reverse order
        early_rul_duration_ = lenth_of_data - early_rul_ # Calculate early RUL duration
        if early_rul_duration_ <= 0:    # This condition is needed when early rul is larger than lenth_of_data of an engine
            target_array = np.arange(lenth_of_data-1, -1, -1) # Create target RUL in reverse order
            return target_array # Return target RUL
        else: # If early RUL duration is greater than 0
            target_array = np.append(early_rul_*np.ones(shape = (early_rul_duration_,)), np.arange(early_rul_-1, -1, -1)) # Create target RUL with early RUL and then in reverse order
            return target_array # Return target RUL

def create_batches_input_and_targets(input_data, target_data_ = None, batch_window_length = 1, shift_among_batches = 1): # Function to create input data batches with corresponding target data
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
    num_of_batches_ = int(np.floor((len(input_data) - batch_window_length)/shift_among_batches)) + 1 # Calculate number of batches
    num_of_features = input_data.shape[1] # Calculate number of features
    output_data_batches = np.repeat(np.nan, repeats =  num_of_batches_ * batch_window_length * num_of_features).reshape( num_of_batches_, batch_window_length,
                                                                                                  num_of_features) # Create output data batches
    if target_data_ is None: # If target data is not provided
        for batch_set in range( num_of_batches_): # Loop through each batch
            output_data_batches[batch_set,:,:] = input_data[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Create input data batches
        return output_data_batches # Return input data batches
    else: # If target data is provided
        output_targets_ = np.repeat(np.nan, repeats =  num_of_batches_) # Create output target data
        for batch_set in range( num_of_batches_):  # Loop through each batch
            output_data_batches[batch_set,:,:] = input_data[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Create input data batches
            output_targets_[batch_set] = target_data_[(shift_among_batches*batch_set + (batch_window_length-1))] # Create target data
        return output_data_batches, output_targets_ # Return input data batches and target data


def processing_test_data(test_data_for_an_engine, batch_window_length, shift_among_batches, num_test_windows = 1):  # Function to process test data for a specific engine and create test data batches
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
    max_num_test_batches_ = int(np.floor((len(test_data_for_an_engine) - batch_window_length)/shift_among_batches)) + 1 # Calculate maximum number of test batches
    if max_num_test_batches_ < num_test_windows: # If maximum number of test batches is less than number of test windows
        required_length  = (max_num_test_batches_ -1)* shift_among_batches + batch_window_length # Calculate required length
        batched_test_data_for_an_engine = create_batches_input_and_targets(test_data_for_an_engine[-required_length :, :],  
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create test data batches
        extracted_num_test_windows = max_num_test_batches_ # Extract number of test windows
        return batched_test_data_for_an_engine, extracted_num_test_windows # Return test data batches and number of test windows
    else: # If maximum number of test batches is greater than number of test windows
        required_length  = (num_test_windows - 1) * shift_among_batches + batch_window_length  # Calculate required length
        batched_test_data_for_an_engine = create_batches_input_and_targets(test_data_for_an_engine[-required_length :, :],
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create test data batches
        extracted_num_test_windows = num_test_windows # Extract number of test windows
        return batched_test_data_for_an_engine, extracted_num_test_windows # Return test data batches and number of test windows



batch_window_length = 1 # Length of each batch
shift_among_batches = 1 # Shift value for creating batches
early_rul_ = 125 # Early RUL value
processed_train_data = [] # List to store processed training data
processed_train_targets = [] # List to store processed training targets
num_test_windows = 5
processed_test_data = []
num_test_windows_list = []

columns_to_be_dropped = [0,1,4] # Columns to be dropped

train_data_first_column = training_data[0] # Extract first column of training data
test_data_first_column = testing_data[0] # Extract first column of testing data

# Scale data for all engines 
scaler = StandardScaler() # Create a StandardScaler object
training_data = scaler.fit_transform(training_data.drop(columns = columns_to_be_dropped)) # Scale training data
testing_data = scaler.transform(testing_data.drop(columns = columns_to_be_dropped)) # Scale testing data

training_data = pd.DataFrame(data = np.c_[train_data_first_column, training_data]) # Create a DataFrame for training data
testing_data = pd.DataFrame(data = np.c_[test_data_first_column, testing_data]) # Create a DataFrame for testing data

num_train_machines = len(training_data[0].unique()) # Number of machines in training data
num_test_machines = len(testing_data[0].unique()) # Number of machines in testing data



# Process trianing data 
for i in np.arange(1, num_train_machines + 1): # Loop through each machine in training data
    temp_train_data = training_data[training_data[0] == i].drop(columns = [0]).values # Extract data for a specific machine

    # Verify if data of given window length can be extracted from training data
    if (len(temp_train_data) < batch_window_length): # If length of data is less than batch window length
        print("Train engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length)) # Print message
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.") # Raise an error

    temp_train_targets = process_targets(lenth_of_data = temp_train_data.shape[0], early_rul_ = early_rul_) # Create target RUL values
    data_for_a_machine, targets_for_a_machine = create_batches_input_and_targets(temp_train_data, temp_train_targets,
                                                                                batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create input data batches with corresponding target data

    processed_train_data.append(data_for_a_machine) # Append processed training data
    processed_train_targets.append(targets_for_a_machine) # Append processed training targets

processed_train_data = np.concatenate(processed_train_data) # Concatenate processed training data
processed_train_targets = np.concatenate(processed_train_targets) # Concatenate processed training targets

# Process test data
for i in np.arange(1, num_test_machines + 1): # Loop through each machine in testing data
    temp_test_data = testing_data[testing_data[0] == i].drop(columns = [0]).values # Extract data for a specific machine

    # Verify if data of given window length can be extracted from test data
    if (len(temp_test_data) < batch_window_length):
        print("Test engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.") # Raise an error

    # Prepare test data
    test_data_for_an_engine, num_windows = processing_test_data(temp_test_data, batch_window_length = batch_window_length, shift_among_batches = shift_among_batches,
                                                             num_test_windows = num_test_windows) # Process test data

    processed_test_data.append(test_data_for_an_engine) # Append processed test data
    num_test_windows_list.append(num_windows) # Append number of test windows

processed_test_data = np.concatenate(processed_test_data)       # Concatenate processed test data
true_rul_ = true_rul_[0].values # Extract true RUL values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets)) # Create a random permutation
processed_train_data, processed_train_targets = processed_train_data[index], processed_train_targets[index] # Shuffle training data

print("wait.............Rul Calculating...............") 

processed_train_data = processed_train_data.reshape(-1, processed_train_data.shape[2]) # Reshape processed training data
processed_test_data = processed_test_data.reshape(-1, processed_test_data.shape[2]) # Reshape processed test data

parameters = {"C":[1, 10, 50, 100],
             'epsilon':[1, 5, 10, 50],
             'kernel':["rbf"]} # Parameters for hyperparameter tuning
 
tuned_svm_reg = GridSearchCV(SVR(),parameters,n_jobs = -1, cv= 10)  # Perform hyperparameter tuning

# tuned_svm_reg.fit(processed_train_data, processed_train_targets)

# tuned_svm_reg.best_params_

best_reg_model = SVR(kernel = "rbf", C = 10, epsilon = 5) # Create a SVR model with best hyperparameters
best_reg_model.fit(processed_train_data, processed_train_targets) # Fit the model on training data
best_reg_model 

rul_pred_tuned = best_reg_model.predict(processed_test_data) # Predict RUL values for test data

preds_for_each_engine_tuned = np.split(rul_pred_tuned, np.cumsum(num_test_windows_list)[:-1]) # Split predicted RUL values for each engine
mean_pred_for_each_engine_tuned = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))
                                   for ruls_for_each_engine, num_windows in zip(preds_for_each_engine_tuned,
                                                                                num_test_windows_list)] # Calculate mean predicted RUL values for each engine
RMSE_tuned = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine_tuned)) # Calculate RMSE
MAE_tuned = mean_absolute_error(true_rul_, mean_pred_for_each_engine_tuned) # Calculate MAE
print("RMSE after hyperparameter tuning: ", RMSE_tuned) # Print RMSE


plt.plot(true_rul_, label = "True RUL", color = "red") # Plot true RUL values
plt.plot(mean_pred_for_each_engine_tuned , label = "Pred RUL", color = "blue")  # Plot predicted RUL values
plt.legend()
plt.show()

indices_of_last_examples = np.cumsum(num_test_windows_list) - 1 # Calculate indices of last examples for each engine
preds_for_last_example = np.concatenate(preds_for_each_engine_tuned)[indices_of_last_examples] # Extract predicted RUL values for last examples

RMSE_new = np.sqrt(mean_squared_error(true_rul_, preds_for_last_example)) # Calculate RMSE for last examples
MAE_new = mean_absolute_error(true_rul_, preds_for_last_example) # Calculate MAE for last examples
print("RMSE (Taking only last example): ", RMSE_new) # Print RMSE

# Plot true and predicted RUL values
plt.plot(true_rul_, label = "True RUL", color = "red")
plt.plot(preds_for_last_example, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()