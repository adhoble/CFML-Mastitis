import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix


def prepData(x,scaler=None):
        """
        Normalize data
        """
        if not scaler:
                scaler = StandardScaler().fit(x)
        x_transformed= scaler.transform(x)
        return x_transformed,scaler

# load data
print ('----- Loading data -----')
data_dir = "../npy_data"
dataset_name = "concat_1000.npy"
data_path = os.path.join(data_dir, dataset_name)
data = np.load (data_path)
print ("Data shape: {}".format(data.shape))
print ('Done Loading data')

# Can only be one of 'patho', 'cow', 'qrt', 'loc'
# Extract y labels
class_target = 'patho'
if class_target == 'patho':
	y = data[:,-4]
elif class_target == 'cow':
	y = data[:,-3]
elif class_target == 'qrt':
	y = data[:, -2]
elif class_target == 'loc':
	y = data[:, -1]
else:
	raise ValueError ("Unrecognized classification target. Abort.")

# Build dataset
X = data[:, :-4]
y = np.asarray(y, dtype=np.int32)

# Construct and Compile Keras Model
def getModel (row_length, num_class):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(512, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(256, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(256, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(128, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(128, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(80, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(80, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(80, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(80, activation='relu', input_dim = row_length))
	model.add(BatchNormalization())
	model.add(Dropout (0.3))

	model.add(Dense(num_class, activation='softmax'))

	model.compile (loss = 'categorical_crossentropy',
		optimizer = 'adam',
		metrics=['accuracy'])
	return model

# directories to save model
model_dir = "trained_models_nn"
if not os.path.exists(model_dir):
	os.makedirs (model_dir)
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
	os.makedirs (checkpoint_dir)
cp_filepath = "checkpoints/model-{epoch:02d}-{val_acc:.2f}.hdf5"

# Get dataset dimensions
row_len = X.shape[1]
num_classes = np.max(y) + 1
# Print model summary
print ("-----Model Summary-----")
print(getModel(row_len, num_classes).summary())

# Train model using 5 fold cross_validation
accs = []
cross_val_n_splits = 5
curr_split = 1
skf = StratifiedKFold(n_splits=cross_val_n_splits, random_state = 12345, shuffle=True)
for train_index, test_index in skf.split(X, y):
	print ('-----Cross Validation {0}/{1}-----'.format(curr_split, cross_val_n_splits))
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]
	Y_train = keras.utils.to_categorical(y_train)
	Y_test = keras.utils.to_categorical(y_test)
	nn_model = getModel(row_len, num_classes)
	cp_callback = ModelCheckpoint (cp_filepath, monitor="val_acc", save_best_only=True)
	nn_model.fit (X_train, Y_train, 
		batch_size = 64,
		epochs = 500, # 1 for testing
		callbacks = [cp_callback],
		validation_data = (X_test, Y_test))
	nn_model.save(os.path.join(model_dir, "cross_val_split{}.h5".format(curr_split)))

	# Evaluate model and get confusion matrix
	y_true = np.argmax(Y_test, axis = -1)
	y_pred = np.argmax(nn_model.predict (X_test, batch_size=1), axis=-1)
	print ("Confusion Matrix, cross val split {}".format(curr_split))
	conf_mat = confusion_matrix (y_true, y_pred)
	print (confusion_matrix(y_true, y_pred))
	conf_mat.dump (os.path.join(model_dir, "cross_val_split{}_cm.npy".format(curr_split)))

	curr_split += 1

print ('----- All done -----')
