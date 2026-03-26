import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Resize images to 224x224 as expected by VGG19
x_train_resized = np.array([tf.image.resize(image, (224, 224)).numpy() for image in x_train])
x_test_resized = np.array([tf.image.resize(image, (224, 224)).numpy() for image in x_test])

# Preprocess images consistently
x_train_resized = preprocess_input(x_train_resized)
x_test_resized = preprocess_input(x_test_resized)

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, 100)
y_test_one_hot = to_categorical(y_test, 100)

# Define the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers for CIFAR-100 classification
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(100, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation with higher intensity
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    datagen.flow(x_train_resized, y_train_one_hot, batch_size=32), 
    epochs=40, 
    validation_data=(x_test_resized, y_test_one_hot), 
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test_resized, y_test_one_hot)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model
model.save('models/vgg19_cifar100.h5')
print("Model saved as models/vgg19_cifar100.h5")


