
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(34)


# Define the path to the input video file
training_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/flight_data2.txt", sep = "\s+", header = None) # Load the training data
testing_data = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/testing_flight_data.txt", sep = "\s+", header = None) # Load the test data
true_rul_ = pd.read_csv("/media/dinesh/Data/UMD/2nd sem/691 industrial AI/Project 4/Project 4/Testing_DataSet/true_rul_1.txt", sep = "\s+", header = None)

def process_targets_(lenth_of_data, early_rul_ = None): # Function to process the target data
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
    else: # If early RUL is provided, create target RUL in reverse order with early RUL value
        early_rul_duration_ = lenth_of_data - early_rul_ # Calculate the duration of early RUL
        if early_rul_duration_ <= 0: # If early RUL duration is less than or equal to 0
            return np.arange(lenth_of_data-1, -1, -1) # Create target RUL in reverse order
        else: # If early RUL duration is greater than 0
            return np.append(early_rul_*np.ones(shape = (early_rul_duration_,)), np.arange(early_rul_-1, -1, -1)) # Create target RUL in reverse order with early RUL value



def create_batches_input_and_targets(input_data_, target_data_ = None, batch_window_length = 1, shift_among_batches = 1): # Function to process the input data with targets
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
    num_of_features = input_data_.shape[1]  # Calculate the number of features
    output_data_batches = np.repeat(np.nan, repeats = num_of_batches * batch_window_length * num_of_features).reshape(num_of_batches, batch_window_length, num_of_features) # Create an array of NaN values
    
    if target_data_ is None: # If target data is not provided
        for batch_set in range(num_of_batches): # Iterate through the number of batches
            output_data_batches[batch_set,:,:] = input_data_[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Assign the input data to the output data
        return output_data_batches # Return the output data
    else:
        output_targets = np.repeat(np.nan, repeats = num_of_batches) # Create an array of NaN values
        for batch_set in range(num_of_batches): # Iterate through the number of batches
            output_data_batches[batch_set,:,:] = input_data_[(0+shift_among_batches*batch_set):(0+shift_among_batches*batch_set+batch_window_length),:] # Assign the input data to the output data
            output_targets[batch_set] = target_data_[(shift_among_batches*batch_set + (batch_window_length-1))] # Assign the target data to the output targets
        return output_data_batches, output_targets # Return the output data and output targets

def processing_test_data(test_data_for_current_engine, batch_window_length, shift_among_batches, num_test_windows = 1): # Function to process the test data
    """
    Processes test data for a specific engine and creates test data batches.

    Parameters:
    - test_data_for_current_engine (numpy.ndarray): Test data for a specific engine.
    - batch_window_length (int): Length of each test data batch_set.
    - shift_among_batches (int): Shift value for creating test data batches.
    - num_test_windows (int, optional): Number of test data windows to take for each engine.

    Returns:
    - batched_test_data_for_an_engine (numpy.ndarray): Processed test data batches.
    - num_test_windows (int): Number of test data windows used.
    """
    max_num_test_batches = int(np.floor((len(test_data_for_current_engine) - batch_window_length)/shift_among_batches)) + 1 # Calculate the maximum number of test batches
    if max_num_test_batches < num_test_windows: # If the maximum number of test batches is less than the number of test windows
        required_len_ = (max_num_test_batches -1)* shift_among_batches + batch_window_length # Calculate the required length
        batched_test_data_for_an_engine = create_batches_input_and_targets(test_data_for_current_engine[-required_len_:, :], 
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Process the input data with targets
        return batched_test_data_for_an_engine, max_num_test_batches  # Return the batched test data for an engine and the maximum number of test batches
    else: # If the maximum number of test batches is greater than or equal to the number of test windows
        required_len_ = (num_test_windows - 1) * shift_among_batches + batch_window_length # Calculate the required length
        batched_test_data_for_an_engine = create_batches_input_and_targets(test_data_for_current_engine[-required_len_:, :],
                                                                          target_data_ = None,
                                                                          batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Process the input data with targets
        return batched_test_data_for_an_engine, num_test_windows # Return the batched test data for an engine and the number of test windows


batch_window_length = 20 # Define the window length
shift_among_batches = 1 # Define the shift_among_batches value
early_rul_ = 125 # Define the early RUL value
processed_train_data_ = [] # Initialize the processed training data
processed_train_targets_ = [] # Initialize the processed training targets

# How many test windows to take for each engine. If set to 1 (this is the default), only last window of test data for
# each engine is taken. If set to a different number, that many windows from last are taken.
# Final output is the average output of all windows.
num_test_windows = 5 # Define the number of test windows
processed_test_data = [] # Initialize the processed test data
num_test_windows_list = [] # Initialize the number of test windows list
columns_to_be_eliminated = [0,1,4] # Define the columns to be dropped

train_data_first_column = training_data[0] # Get the first column of the training data
test_data_first_column = testing_data[0] # Get the first column of the test data

# Scale data for all engines
scaler = MinMaxScaler(feature_range = (-1,1)) # Initialize the MinMaxScaler
training_data = scaler.fit_transform(training_data.drop(columns = columns_to_be_eliminated)) # Fit and transform the training data
testing_data = scaler.transform(testing_data.drop(columns = columns_to_be_eliminated)) # Transform the test data

training_data = pd.DataFrame(data = np.c_[train_data_first_column, training_data]) # Create a DataFrame for the training data
testing_data = pd.DataFrame(data = np.c_[test_data_first_column, testing_data]) # Create a DataFrame for the test data

num_train_machines = len(training_data[0].unique()) # Get the number of unique training machines
num_test_machines = len(testing_data[0].unique()) # Get the number of unique test machines



# Process training data
for i in np.arange(1, num_train_machines + 1): # Iterate through the number of training machines
    temp_train_data = training_data[training_data[0] == i].drop(columns = [0]).values # Get the training data for a specific machine

    # Verify if data of given window length can be extracted from training data
    if (len(temp_train_data) < batch_window_length): # If the length of the training data is less than the window length
        print("Train engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length)) # Print the message
        raise AssertionError("Window length is larger than the number of data points for some engines. Try decreasing window length.") # Raise an AssertionError

    temp_train_targets = process_targets_(lenth_of_data = temp_train_data.shape[0], early_rul_ = early_rul_) # Process the target data
    data_for_a_machine, targets_for_a_machine = create_batches_input_and_targets(temp_train_data, temp_train_targets,    # Process the input data with targets
                                                                                batch_window_length = batch_window_length, shift_among_batches = shift_among_batches) # Define the window length and shift_among_batches

    processed_train_data_.append(data_for_a_machine) # Append the data for a machine
    processed_train_targets_.append(targets_for_a_machine) # Append the targets for a machine

processed_train_data_ = np.concatenate(processed_train_data_) # Concatenate the processed training data
processed_train_targets_ = np.concatenate(processed_train_targets_) # Concatenate the processed training targets

# Process test data
for i in np.arange(1, num_test_machines + 1): # Iterate through the number of test machines
    temp_test_data = testing_data[testing_data[0] == i].drop(columns = [0]).values # Get the test data for a specific machine

    # Verify if data of given window length can be extracted from test data
    if (len(temp_test_data) < batch_window_length): # If the length of the test data is less than the window length
        print("Test engine {} doesn't have enough data for batch_window_length of {}".format(i, batch_window_length))
        raise AssertionError("Window length is larger than the number of data points for some engines. Try decreasing window length.") # Raise an AssertionError

    # Prepare test data
    test_data_for_current_engine, num_windows = processing_test_data(temp_test_data, batch_window_length = batch_window_length, shift_among_batches = shift_among_batches,
                                                             num_test_windows = num_test_windows) # Process the test data

    processed_test_data.append(test_data_for_current_engine) # Append the test data for an engine
    num_test_windows_list.append(num_windows) # Append the number of test windows

processed_test_data = np.concatenate(processed_test_data) # Concatenate the processed test data
true_rul_ = true_rul_[0].values # Get the true RUL values

# Shuffle training data
index = np.random.permutation(len(processed_train_targets_)) # Create a random permutation of the training targets
processed_train_data_, processed_train_targets_ = processed_train_data_[index], processed_train_targets_[index] # Shuffle the training data and targets


processed_train_data_, processed_val_data, processed_train_targets_, processed_val_targets = train_test_split(processed_train_data_,
                                                                                                            processed_train_targets_,
                                                                                                            test_size = 0.2,
                                                                                                            random_state = 83) # Split the training data into training and validation data
def create_compiled_model(): # Function to create a compiled model
    """
    Creates a compiled CNN model.

    Returns:
    - model (tensorflow.keras.models.Sequential): Compiled CNN model.
    """
    model = Sequential([
        layers.Conv1D(256, 7, activation = "relu", input_shape = (batch_window_length, processed_train_data_.shape[2])), # Create a 1D convolutional layer
        layers.Conv1D(96, 7, activation = "relu"), # Create a 1D convolutional layer
        layers.Conv1D(32, 7, activation = "relu"), # Create a 1D convolutional layer
        layers.GlobalAveragePooling1D(), # Create a global average pooling layer
        layers.Dense(64, activation = "relu"), # Create a dense layer
        layers.Dense(128, activation = "relu"), # Create a dense layer
        layers.Dense(1) # Create a dense layer
    ])
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)) # Compile the model
    return model

def scheduler(epoch): # Function to schedule the learning rate
    """
    Learning rate scheduler for the model training.

    Parameters:
    - epoch (int): Current epoch number.

    Returns:
    - learning_rate (float): Learning rate for the current epoch.
    """
    if epoch < 10: # If the epoch is less than 10
        return 0.001 # Return the learning rate
    else: # If the epoch is greater than or equal to 10
        return 0.0001 # Return the learning rate

callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1) # Define the learning rate scheduler callback

model = create_compiled_model() # Create the compiled model
history = model.fit(processed_train_data_, processed_train_targets_, epochs = 30, 
                    validation_data = (processed_val_data, processed_val_targets),
                    callbacks = callback,
                    batch_size = 64, verbose = 2) # Fit the model

rul_pred = model.predict(processed_test_data).reshape(-1) # Predict the RUL values
preds_for_each_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1]) # Split the RUL predictions for each engine
mean_pred_for_each_engine = [np.average(ruls_for_each_engine, weights = np.repeat(1/num_windows, num_windows))  # Calculate the mean RUL prediction for each engine
                             for ruls_for_each_engine, num_windows in zip(preds_for_each_engine, num_test_windows_list)] # Iterate through the RUL predictions and number of test windows

mean_pred_for_each_engine = mean_pred_for_each_engine[:-1] # Remove the last element from the mean prediction
true_rul_ = true_rul_[:-1] # Remove the last element from the true RUL
RMSE = np.sqrt(mean_squared_error(true_rul_, mean_pred_for_each_engine)) # Calculate the RMSE
print("RMSE for when prediction made by mean of last 5 samples: ", RMSE) # Print the RMSE

# Plot true and predicted RUL values
plt.plot(true_rul_, label = "True RUL", color = "red") # Plot the true RUL values
plt.plot(mean_pred_for_each_engine, label = "Pred RUL", color = "blue")
plt.legend()
plt.show()

tf.keras.models.save_model(model, "CNN_piecewise_RMSE_"+ str(np.round(RMSE, 4)) + ".h5") # Save the model
 
indices_of_last_examples = np.cumsum(num_test_windows_list) - 1 # Get the indices of the last examples
preds_for_last_example = np.concatenate(preds_for_each_engine)[indices_of_last_examples] # Get the predictions for the last example


preds_for_last_example = preds_for_last_example[:-1] # Remove the last element from the predictions
# true_rul_ = true_rul_[:-1] # Remove the last element from the true RUL
RMSE_new = np.sqrt(mean_squared_error(true_rul_, preds_for_last_example)) # Calculate the RMSE
print("RMSE for when prediction made by last 1 samples ", RMSE_new) # Print the RMSE


# Plot true and predicted RUL values
plt.plot(true_rul_, label = "True RUL", color = "red") # Plot the true RUL values
plt.plot(preds_for_last_example, label = "Pred RUL", color = "blue") # Plot the predicted RUL values
plt.legend()
plt.show()

