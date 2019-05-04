import keras
from keras.datasets import cifar10
from keras.models import Sequential
import keras.layers as layers
import os

# some set variabes
batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20
# input parser here
activation = 'relu'

save_dir = os.path.join(os.getcwd(), 'model_save')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("data shape: {}".format(x_train.shape))

#y_train = keras.

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

""" Taking this out for first test
model.add(layers.Conv2D(128,(3,3),padding='same'))
model.add(layers.Activation(activation))
model.add(layers.MaxPooling2D(pool_size=(4,4)))
model.add(layers.Dropout(0.25))
"""

model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation(activation))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001,decay=1e-6)

model.compile(loss='categorical_crossentropy',optimier=opt,metrics=['accuracy'])

# Train the model
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test),shuffle=True)

# save model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# see what score it gets
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])