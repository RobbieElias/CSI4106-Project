# CSI4106 - Group Project
# Written by: Vincent Arrage (8579139), Maxime Bassett (8716035), Robbie Elias (7953896)
# 
# Train a model for classifying different hand gestures using Transfer Learning
# on MobileNet V2, Inception V3, and Inception-ResNet V2.
#
# Using some code and recommendations provided by TensorFlow at:
# https://www.tensorflow.org/lite/models/image_classification/overview

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

class Model:
    MOBILE_NET_V2 = 1
    INCEPTION_V3 = 2
    INCEPTION_RES_NET_V2 = 3

# the base model we're using to train (change this if needed)
# Use Model.MOBILE_NET_V2 for baseline
BASE_MODEL = Model.MOBILE_NET_V2

IMAGE_SIZE = 224 # Image size is 224x224
BATCH_SIZE = 16
DIRECTORY = 'images' # Where the images are stored

# Generator that will transform the image dataset to get a wider variety of data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    fill_mode="nearest",
    validation_split=0.2
)

# Use generator above to obtain training data
print("Training Data:")
train_generator = datagen.flow_from_directory(
    DIRECTORY,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True, # Make sure to shuffle the data
    subset='training')

# Use generator above to obtain validation data
print("Validation Data:")
val_generator = datagen.flow_from_directory(
    DIRECTORY,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    shuffle=True, # Make sure to shuffle the data
    subset='validation')

# Write the labels to a "labels.txt" file
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
	f.write(labels)

# Print the labels
print("\nClass Labels:")
print(labels)
print()

# Define the image shape (3 is for RGB)
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

if (BASE_MODEL == Model.MOBILE_NET_V2):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
    	input_shape=IMG_SHAPE,
    	include_top=False, 
    	weights='imagenet')

    print('Training using pre-trained model: MobileNet V2')

    # Setup the configuration for this model
    dropout = 0.2
    pooling = tf.keras.layers.GlobalAveragePooling2D()
    optimizer = tf.keras.optimizers.Adam()
    model_name = 'model_mobilenet.tflite'
elif (BASE_MODEL == Model.INCEPTION_V3):
    # Create the base model from the pre-trained model Inception V3
    base_model = tf.keras.applications.InceptionV3(
        input_shape=IMG_SHAPE,
        include_top=False, 
        weights='imagenet')

    print('Training using pre-trained model: Inception V3')

    # Setup the configuration for this model
    dropout = 0.25
    pooling = tf.keras.layers.GlobalMaxPooling2D()
    optimizer = tf.keras.optimizers.Adam()
    model_name = 'model_inception.tflite'
elif (BASE_MODEL == Model.INCEPTION_RES_NET_V2):
    # Create the base model from the pre-trained model Inception-ResNet V2
    base_model = tf.keras.applications.InceptionResNetV2(
        input_shape=IMG_SHAPE,
        include_top=False, 
        weights='imagenet')

    print('Training using pre-trained model: Inception-ResNet V2')

    # Setup the configuration for this model
    dropout = 0.25
    pooling = tf.keras.layers.GlobalAveragePooling2D()
    optimizer = tf.keras.optimizers.Nadam()
    model_name = 'model_resnet.tflite'

# Freeze the convolutional base
base_model.trainable = False

# Create a sequential model, based off the base model defined above
model = tf.keras.Sequential([
	base_model,
	tf.keras.layers.Conv2D(32, 3, activation='relu'),
	tf.keras.layers.Dropout(dropout),
	pooling,
	tf.keras.layers.Dense(len(train_generator.class_indices.keys()), activation='softmax')
])

# Compile the model (use categorical crossentropy since more than two classes)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Number of epochs, more might be better but takes longer
epochs = 10

# Train the model using the training generator, and validate using validation generator
history = model.fit_generator(train_generator, epochs=epochs, validation_data=val_generator)

# Save the model
saved_model_dir = 'model'
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the tflite model file
with open(model_name, 'wb') as f: f.write(tflite_model)


# Plotting The data
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
