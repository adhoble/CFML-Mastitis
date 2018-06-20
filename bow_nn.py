import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold
#from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def prepData(x,scaler=None):
        """
        Normalize data
        """
        if not scaler:
                scaler = StandardScaler().fit(x)
        x_transformed= np.nan_to_num(np.array(scaler.transform(x)))
        return x_transformed,scaler


# load data
print ('----- Loading data -----')
data_dir = "/home/n-z/plahiri2/mastitis/large_cells"
dataset_name = "bow_200words_qrt.npy"
data_path = os.path.join(data_dir, dataset_name)
data = np.load (data_path)
print ("Data shape: {}".format(data.shape))
print ('Done Loading data')

# Can only be one of 'patho', 'cow', 'qrt', 'loc'
# Extract y labels
class_target = 'patho'

# "classifier" for label predictions or "regressor" for intensity regression
model_type = "classifier"

if class_target == 'patho':
	y = data[:,-6]
elif class_target == 'cow':
	y = data[:,-5]
elif class_target == 'qrt':
	y = data[:, -4]
elif class_target == 'loc':
	y = data[:, -3]
elif class_target == 'lact':
	y = data[:, -2]
elif class_target == 'int':
	y = data[:, -1]
	model_type = 'regressor'
elif class_target == 'int_class':
	intensities = data[:,-1]
	n = len(intensities)
	int_encoder = {}
	y = np.zeros(n)
	curr_index = 0
	for i in range(n):
		res = int_encoder.get(intensities[i], None)
		if res is None:
			int_encoder[intensities[i]] = curr_index
			curr_index += 1
			y[i] = curr_index
		else:
			y[i] = int_encoder[intensities[i]]

else:
	raise ValueError ("Unrecognized classification target. Abort.")


# Build dataset
"""
n, _ = data.shape
perm = np.random.permutation(n)
data = data[perm]
y = y[perm]
split_point = int(0.8 * n)
X = data[:split_point, :-4]
y = np.asarray(y, dtype=np.int32)
temp = y[:split_point]

X_test = data[split_point:, :-4]
y_test = y[split_point:]
print ("Test set healthy ratio: {}".format(np.sum(y_test == 1)/len(y_test)))

y = temp
"""
X = data[:, :-6]
# Construct and Compile Keras Model
def getModelClassifier (row_length, num_class, dropout_rate=0.2, optimizer='adam'):
	model = Sequential()
	model.add(Dense(256, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(128, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(64, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(num_class, activation='softmax'))

	model.compile (loss = 'categorical_crossentropy',
		optimizer = optimizer,
		metrics=['accuracy'])
	return model

def getModelRegressor(row_length, dropout_rate=0.2, optimizer='adam'):
	model = Sequential()
	model.add(Dense(256, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(128, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(64, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (dropout_rate))

	model.add(Dense(1))

	model.compile (loss = 'mse',
		optimizer = optimizer,
		metrics=['mse'])
	return model

# directories to save model
model_dir = "trained_models"
if not os.path.exists(model_dir):
	os.makedirs (model_dir)
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
	os.makedirs (checkpoint_dir)
cp_filepath = "checkpoints/model-{epoch:02d}-{val_acc:.2f}.hdf5"

# Get dataset dimensions
row_len = X.shape[1]
if model_type == 'classifier':
	num_classes = int(np.max(y) + 1)

if model_type == 'classifier':
	y = keras.utils.to_categorical(y)

# Train model using 5 fold cross_validation
"""
accs = []
cross_val_n_splits = 5
curr_split = 1
skf = StratifiedKFold(n_splits=cross_val_n_splits, random_state = 12345, shuffle=True)
for train_index, test_index in skf.split(X, y):
	print ('-----Cross Validation {0}/{1}-----'.format(curr_split, cross_val_n_splits))
	X_train, y_train = X[train_index], y[train_index]
	X_val, y_val = X[test_index], y[test_index]
	Y_train = keras.utils.to_categorical(y_train)
	Y_val = keras.utils.to_categorical(y_val)
	nn_model = getModel(row_len, num_classes)
	cp_callback = ModelCheckpoint (cp_filepath, monitor="val_acc", save_best_only=True)
	nn_model.fit (X_train, Y_train, 
		batch_size = 128,
		epochs = 100, # 1 for testing
		callbacks = [cp_callback],
		validation_data = (X_val, Y_val))
	nn_model.save(os.path.join(model_dir, "cross_val_split{}.h5".format(curr_split)))

	# Evaluate model and get confusion matrix
	y_true = y_test
	y_pred = np.argmax(nn_model.predict (X_test, batch_size=1), axis=-1)
	print ("Confusion Matrix, cross val split {}".format(curr_split))
	conf_mat = confusion_matrix (y_true, y_pred)
	print (confusion_matrix(y_true, y_pred))
	conf_mat.dump (os.path.join(model_dir, "cross_val_split{}_cm.npy".format(curr_split)))
	print ("Test accuracy: {}".format(np.sum(np.diag(conf_mat))/np.sum(conf_mat)))
	curr_split += 1
"""

# Create sklearn wrapped model
model = None
if model_type == 'classifier':
	model = KerasClassifier(build_fn = getModelClassifier, row_length=row_len, num_class=num_classes)
elif model_type == 'regressor':
	model = KerasRegressor(build_fn = getModelRegressor, row_length=row_len)

# Train, validate and test the model using Nested CrossValidation
"""
params = {"dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5], 
          "epochs": [100],
          "batch_size": [64, 128],
          "optimizer": ["adam"],
          "verbose": [2]}
"""

params = {"dropout_rate": [0.1], 
          "epochs": [30],
          "batch_size": [64]}


models = []
test_accs = np.zeros(0)
test_losses = np.zeros(0)
for tr_index, ts_index in KFold(n_splits=4, shuffle=True).split(X, y):
	grid_model = GridSearchCV (model, params)
	#print (y[tr_index])
	X_train, scaler = prepData(X[tr_index])
	grid_model.fit(X_train, y[tr_index])
	best_model = grid_model.best_estimator_
	models.append(best_model)
	X_test, _ = prepData(X[ts_index], scaler)
	y_test = y[ts_index]
	if model_type == 'classifier':
		y_true = np.argmax(y_test, axis=1)
		y_pred = best_model.predict(X_test, batch_size=1)
		conf_mat = confusion_matrix(y_true, y_pred)
		print (conf_mat)
		test_acc = np.sum(np.diag(conf_mat))/np.sum(conf_mat)
		test_accs = np.append(test_accs, test_acc)
	else:
		y_pred = best_model.predict(X_test, batch_size=1)
		# Compute mean square loss
		mean_sqr_err = np.sum(np.square(y_test - y_pred)) / len(y_test)
		test_losses = np.append(test_losses, mean_sqr_err)

if model_type == 'classifier':
	print ("Test Accuracy: {0} +- {1}".format(np.mean(test_accs), np.std(test_accs)))
else:
	print ("Test Loss: {0} +- {1}".format(np.mean(test_losses), np.std(test_losses)))

# Save models
for i in range(len(models)):
	if model_type == 'classifier':
		models[i].model.save(os.path.join(model_dir, "model_{0}_{1}.h5".format(i, test_accs[i])))
	else:
		models[i].model.save(os.path.join(model_dir, "model_{0}_{1}.h5".format(i, test_losses[i])))
if model_type == 'classifier':
	best_index = np.argmax(test_accs)
else:
	best_index = np.argmin(test_losses)
best_model_keras = models[best_index].model

# Print summary of the best model
print ("Summary of model {0}: {1}".format(best_index,best_model_keras.summary()))

if class_target == 'int_class':
	print ("Intensity value encoder:")
	print (int_encoder)

# Plot the structure of the best model for illustration
#plot_model (best_model_keras, to_file="best_model_structure.png", show_shapes=True)
print ('----- All done -----')
