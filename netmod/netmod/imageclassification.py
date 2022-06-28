import os

import numpy as np
import cv2
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# read file
def readDirectory(dir):
    L = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")
        L = L + len(filenames)

#ImageProcessing
# Grayscale
def rgb2gray(rgb_img):
    output_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    return output_img

# Scale to 0 to 1
def scale(image):
    return image / 255

#resize
def resize_img(image, rows=224, cols=224):
    return cv2.resize(image, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)

# resize the shape
def reshape(image, axis):
    return np.expand_dims(image.mean(axis=axis), axis=1)

# Function to call other Preprocessing Functions
def preprocessed_img(input_img):
    output_img = rgb2gray(input_img)
    output_img = scale(output_img)
    output_img = resize_img(output_img)
    output_img = reshape(output_img, 1)
    return output_img

#data preprocessing
def ann(dir,e):
    L = 0
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")
        L = L + len(filenames)


    # matrix containing L vectors of shape (224, 1)
    mean_vector_matrix = np.zeros(shape=(L, 224, 1))
    # Target vector containing the classes for L images
    target_vector = np.zeros(shape=(L, 1))

    n = 0
    for root, dirnames, filenames in os.walk(dir+r"\1"):   
        n_total = len(filenames)
        for filename in filenames:
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)
            img = preprocessed_img(img)
            mean_vector_matrix[n] = img
            target_vector[n] = 1
            if n % 20 == 0:
                print(f"File {n} {filename}")
            n = n + 1  

    for root, dirnames, filenames in os.walk(dir+r"\0"):
        n_total = len(filenames)
        for filename in filenames:
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)
            img = preprocessed_img(img)
            mean_vector_matrix[n] = img
            target_vector[n] = 0
            if n % 20 == 0:
                print(f"File {n} {filename}")
            n = n + 1  

    SEED = 2
    x_train, x_test, y_train, y_test = train_test_split(mean_vector_matrix, target_vector, test_size=0.1, random_state=SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    input_shape = x_train.shape
    print(input_shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu', input_shape = input_shape),
        tf.keras.layers.Dense(units=64, activation='relu', input_shape = input_shape),
        tf.keras.layers.Dense(units=64, activation='relu', input_shape = input_shape),
        tf.keras.layers.Dense(units=64, activation='relu', input_shape = input_shape),
        tf.keras.layers.Dense(units=2, activation='softmax')
        ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer = 'sgd',
        metrics = ['accuracy']
        )

    history = model.fit(
        x = x_train,
        y = y_train,
        epochs = e
    )

    model.save('model.h5')

def cnn(dir,e):
    # Defining batch specfications
    img_height = 256
    img_width = 256
    seed = 42

    # loading training set
    training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir+r'\training',
    seed=seed,
    image_size= (img_height, img_width))

    # loading validation dataset
    validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    dir+r'\validation',
    seed=seed,
    image_size= (img_height, img_width))

    class_names = training_ds.class_names

    # Configuring dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Defining Cnn
    model = tf.keras.models.Sequential([
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(len(class_names), activation= 'softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(training_ds, validation_data = validation_ds, epochs = e)

    model.save('model.h5')