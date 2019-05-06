import keras
from keras.datasets import cifar10
from keras.models import Sequential
import keras.layers as layers
import os
import sys
import time

# some set variabes
batch_size = 32
num_classes = 10
epochs = 200
num_predictions = 20
# input parser here
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
if not sys.argv[1]:
	print("Please provide an argument in range 0-9 for activation function.")
	sys.exit()
selection = sys.argv[1]
if isinstance(selection,str):
	selection = int(selection)

# This will be our activation function
activation = activation_map.get(selection)
print("Using {} activation function for everything except the last layer.".format(activation))


save_dir = os.path.join(os.getcwd(), 'model_save')
model_name = "cifar10_{}.h5".format(activation)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("data shape: {}".format(x_train.shape))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Making the model
model = Sequential()
model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model.add(layers.Activation(activation))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,(3,3),padding='same'))
model.add(layers.Activation(activation))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

# this is something added in addition to the number of layers in the example
model.add(layers.Conv2D(128,(3,3),padding='same'))
model.add(layers.Activation(activation))
model.add(layers.MaxPooling2D(pool_size=(4,4)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation(activation))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001,decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

# Normalize data and data types
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Timing
start = time.time()
# Train the model
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),shuffle=True,verbose=0)
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
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('------------------------------------------------')