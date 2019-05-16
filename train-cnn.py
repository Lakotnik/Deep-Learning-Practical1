import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, Dropout
import os
import sys
import time

activation_map = {
	0 : "relu",
	1 : "softmax",
	2 : "elu",
	3 : "softplus",
	4 : "softsign",
	5 : "tanh",
	6 : "sigmoid",
	7 : "hard_sigmoid",
	8 : "exponential",
	9 : "linear"
}

class ConvNet():

	def __init__(self,activation="relu",input_shape=(32,32,3),dropout=0.3):
		self.input_shape = input_shape
		self.dropout     = dropout
		self.activation  = activation
		self.batch_size  = 32
		self.num_classes = 10
		self.epochs      = 50
		self.optimizer   = 'adam'
		self.metrics     = ['accuracy']
		self.model       = Sequential()

	def create_model(self):
		self.model.add(Conv2D(32, kernel_size=3, activation=self.activation, input_shape=self.input_shape))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(32, kernel_size=3, activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(self.dropout))

		self.model.add(Conv2D(64, kernel_size=3, activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(64, kernel_size=3, activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(self.dropout))

		# Final bit
		self.model.add(Flatten())
		self.model.add(Dense(128, activation=self.activation))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(self.dropout))
		self.model.add(Dense(self.num_classes, activation='softmax'))

		self.model.compile(optimizer=self.optimizer, loss="categorical_crossentropy", metrics=self.metrics)

		return self.model

if not sys.argv[1]:
	print("Please provide an argument in range 0-9 for activation function.")
	sys.exit()
selection = sys.argv[1]
if isinstance(selection,str):
	selection = int(selection)

# This will be our activation function
activation = activation_map.get(selection)
print("Using {} activation function for everything except the last layer.".format(activation))

# Save directory
save_dir = os.path.join(os.getcwd(), 'model_save')
model_name = "cifar10_{}.h5".format(activation)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("data shape: {}".format(x_train.shape))

# create model
generator = ConvNet(activation=activation,input_shape=x_train.shape[1:],dropout=0.4)
model = generator.create_model()

# Format data
y_train = keras.utils.to_categorical(y_train, generator.num_classes)
y_test = keras.utils.to_categorical(y_test, generator.num_classes)

# Normalize data and data types
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Timing
start = time.time()
# Train the model
model.fit(x_train,y_train,batch_size=generator.batch_size,epochs=generator.epochs,validation_data=(x_test,y_test),shuffle=True,verbose=0)
end = time.time()


# save model
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# see what score it gets
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('------------------------------------------------')
print('Training took {} seconds.'.format((end-start)))
print('Activation: ',activation)
print('Dropout: ',generator.dropout)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('------------------------------------------------')