
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(34)


# Define the path to the input video file

training_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/flight_data2.txt", sep = "\s+", header = None) # Load the training data
testing_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/testing_flight_data.txt", sep = "\s+", header = None) # Load the testing data
true_rul_ = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/true_rul_1.txt", sep = "\s+", header = None) # Load the true RUL values


def process_targets(lenth_of_data, early_rul_ = None): # Function to process the targets
    """
    Takes data length and early RUL (Remaining Useful Life) as input and creates target RUL.

    Parameters:
    - lenth_of_data (int): Length of the data.
    - early_rul_ (int, optional): Early RUL value. If None, target RUL is created in reverse order.

    Returns:
    - target_rul (numpy.ndarray): Array of target RUL values.
    """
    if early_rul_ == None: # If early RUL is not provided
        return np.arange(lenth_of_data-1, -1, -1) # Return the target RUL values in reverse order
    else:
        early_rul_duration = lenth_of_data - early_rul_ # Calculate the early RUL duration
        if early_rul_duration <= 0: # If early RUL duration is less than or equal to 0
            return np.arange(lenth_of_data-1, -1, -1) # Return the target RUL values in reverse order
        else: # If early RUL duration is greater than 0
            return np.append(early_rul_*np.ones(shape = (early_rul_duration,)), np.arange(early_rul_-1, -1, -1)) # Return the target RUL values


def create_batches_input_and_targets(input_data_, target_data_ = None, batch_window_length = 1, shift_among_batches = 1): # Function to create input data batches with corresponding target data
    """
    Processes input data and creates input data batches with corresponding target data.

    Parameters:
    - input_data_ (numpy.ndarray): Input data.
    - target_data_ (numpy.ndarray, optional): Target data. If None, only input data batches are created.
    - batch_window_length (int, optional): Length of each input data batch_set.
    - shift_among_batches (int, optional): Shift value for creating input data batches.

    Returns:
    - output_data_batches (numpy.ndarray): Processed input data batches.
    - output_targets (numpy.ndarray, optional): Processed target data.
    """
    num_of_batches = int(np.floor((len(input_data_) - batch_window_length)/shift_among_batches)) + 1 # Calculate the number of batches
    num_of_features = input_data_.shape[1] # Calculate the number of features
    output_data_batches = np.repeat(np.nan, repeats = num_of_batches * batch_window_length * num_of_features).reshape(num_of_batches, batch_window_length,
                                                                                                  num_of_features) # Create an array of NaN values
    if target_data_ is None: # If target data is not provided
        for batch_set in range(num_of_batches): # For each batch set
            output_data_batches[batch_set,:,:] = input_data_[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Assign the input data to the output data batches
        return output_data_batches # Return the output data batches
    else: # If target data is provided
        output_targets = np.repeat(np.nan, repeats = num_of_batches) # Create an array of NaN values
        for batch_set in range(num_of_batches): # For each batch set
            output_data_batches[batch_set,:,:] = input_data_[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Assign the input data to the output data batches
            output_targets[batch_set] = target_data_[(shift_among_batches*batch_set + (batch_window_length-1))] # Assign the target data to the output targets
        return output_data_batches, output_targets # Return the output data batches and the output targets



def processing_test_data(test_data_for_an_engine, batch_window_length, shift_among_batches, num_test_windows = 1): # Function to process the test data
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
    max_num_test_batches_ = int(np.floor((len(test_data_for_an_engine) - batch_window_length)/shift_among_batches)) + 1 # Calculate the maximum number of test batches
    if max_num_test_batches_ < num_test_windows: # If the maximum number of test batches is less than the number of test windows
        required_length = (max_num_test_batches_ -1)* shift_among_batches + batch_window_length # Calculate the required length
        batched_test_data_for_an_engine_ = create_batches_input_and_targets(test_data_for_an_engine[-required_length:, :], 
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create test data batches
        return batched_test_data_for_an_engine_, max_num_test_batches_ # Return the test data batches and the maximum number of test batches
    else: # If the maximum number of test batches is greater than or equal to the number of test windows
        required_length = (num_test_windows - 1) * shift_among_batches + batch_window_length # Calculate the required length
        batched_test_data_for_an_engine_ = create_batches_input_and_targets(test_data_for_an_engine[-required_length:, :],  
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create test data batches
        return batched_test_data_for_an_engine_, num_test_windows # Return the test data batches and the number of test windows

batch_window_length = 20 # Define the length of each batch window
shift_among_batches = 1 # Define the shift value for creating batches
early_rul_ = 125 # Define the early RUL value
processed_training_data = [] # Initialize the processed training data
processed_train_targets = [] # Initialize the processed training targets


num_test_windows = 5 # Define the number of test windows
processed_testing_data = [] # Initialize the processed testing data
num_test_windows_list = [] # Initialize the number of test windows list


columns_to_be_eliminated  = [0,1,4] # Define the columns to be eliminated

training_data_first_column = training_data[0] # Extract the first column of the training data
test_data_first_column = testing_data[0] # Extract the first column of the testing data

# Scale data for all engines
scaler = StandardScaler() # Initialize the standard scaler
training_data = scaler.fit_transform(training_data.drop(columns = columns_to_be_eliminated )) # Scale the training data
testing_data = scaler.transform(testing_data.drop(columns = columns_to_be_eliminated )) # Scale the testing data
 
training_data = pd.DataFrame(data = np.c_[training_data_first_column, training_data]) # Create a data frame for the training data
testing_data = pd.DataFrame(data = np.c_[test_data_first_column, testing_data]) # Create a data frame for the testing data

num_train_machines_ = len(training_data[0].unique()) # Calculate the number of training machines
num_test_machines_ = len(testing_data[0].unique()) # Calculate the number of testing machines


# Process trianing data
for i in np.arange(1, num_train_machines_ + 1): # For each training machine
    temp_training_data = training_data[training_data[0] == i].drop(columns = [0]).values # Extract the training data for the current machine

    # Verify if data of given window length can be extracted from training data
    if (len(temp_training_data) < batch_window_length): # If the length of the training data is less than the batch window length
        print("Train engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length))
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.") # Raise an assertion error

    temp_train_targets = process_targets(lenth_of_data = temp_training_data.shape[0], early_rul_ = early_rul_) # Process the targets for the current machine
    data_for_a_machine, targets_for_a_machine = create_batches_input_and_targets(temp_training_data, temp_train_targets,
                                                                                batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Create input data batches with corresponding target data
 
    processed_training_data.append(data_for_a_machine) # Append the input data batches to the processed training data
    processed_train_targets.append(targets_for_a_machine) # Append the target data to the processed training targets

processed_training_data = np.concatenate(processed_training_data) # Concatenate the processed training data
processed_train_targets = np.concatenate(processed_train_targets) # Concatenate the processed training targets

# Process test data
for i in np.arange(1, num_test_machines_ + 1): # For each testing machine
    temp_testing_data = testing_data[testing_data[0] == i].drop(columns = [0]).values # Extract the testing data for the current machine

    # Verify if data of given window length can be extracted from test data
    if (len(temp_testing_data) < batch_window_length): # If the length of the testing data is less than the batch window length
        print("Test engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length)) # Print a message
        raise AssertionError("Window length is larger than number of data points for some engines. "
                             "Try decreasing window length.") # Raise an assertion error

    # Prepare test data
    test_data_for_an_engine, num_windows = processing_test_data(temp_testing_data, batch_window_length = batch_window_length, shift_among_batches = shift_among_batches,
                                                             num_test_windows = num_test_windows) # Process the test data for the current machine

    processed_testing_data.append(test_data_for_an_engine) # Append the test data to the processed testing data
    num_test_windows_list.append(num_windows) # Append the number of test windows to the number of test windows list

processed_testing_data = np.concatenate(processed_testing_data) # Concatenate the processed testing data
true_rul_ = true_rul_[0].values # Extract the true RUL values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets)) # Generate a random permutation 
processed_training_data, processed_train_targets = processed_training_data[index], processed_train_targets[index] # Shuffle the processed training data and the processed training targets



processed_training_data, processed_val_data, processed_train_targets, processed_val_targets = train_test_split(processed_training_data,
                                                                                                            processed_train_targets,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 83) # Split the training data into training and validation data


def create_compiled_model(): # Function to create a compiled model
    model = Sequential([
        layers.LSTM(128, input_shape = (batch_window_length, 5), return_sequences=True, activation = "tanh"), # Create an LSTM layer
        layers.LSTM(64, activation = "tanh", return_sequences = True), # Create an LSTM layer
        layers.LSTM(32, activation = "tanh"), # Create an LSTM layer
        layers.Dense(96, activation = "relu"), # Create a dense layer
        layers.Dense(128, activation = "relu"), # Create a dense layer
        layers.Dense(1) # Create a dense layer
    ])
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)) # Compile the model
    return model # Return the model
 
def scheduler(epoch): # Function to schedule the learning rate
    if epoch < 5: # If the epoch is less than 5
        return 0.001 
    else: # If the epoch is greater than or equal to 5
        return 0.0001

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1) # Define the learning rate scheduler

model = create_compiled_model() # Create a compiled model
history = model.fit(processed_training_data, processed_train_targets, epochs = 10, 
                    validation_data = (processed_val_data, processed_val_targets),
                    callbacks = callback,
                    batch_size = 128, verbose = 2) # Fit the model

rul_pred = model.predict(processed_testing_data).reshape(-1) # Predict the RUL values for the test data
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1]) # Split the predicted RUL values for each engine
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows)) 
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)] # Calculate the mean predicted RUL values for each engine
RMSE = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine)) # Calculate the RMSE
print("RMSE: ", RMSE) # Print the RMSE

# Plot true and predicted RUL values
plt.plot(true_rul_, label = "True RUL", color = "red")      
plt.plot(mean_pred_for_each_engine, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()

tf.keras.models.save_model(model, "LSTM_RUL_PREDICTION"+ str(np.round(RMSE, 4)) + ".h5") # Save the model

indices_of_last_examples = np.cumsum(num_test_windows_list) - 1 # Calculate the indices of the last examples
preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples]  # Extract the predicted RUL values for the last examples

RMSE_new = np.sqrt(mean_squared_error(true_rul_, preds_for_last_example)) # Calculate the RMSE for the last examples
print("RMSE (Taking only last examples): ", RMSE_new) # Print the RMSE for the last examples


# Plot true and predicted RUL values
plt.plot(true_rul_, label = "True RUL", color = "red")
plt.plot(preds_for_last_example, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()