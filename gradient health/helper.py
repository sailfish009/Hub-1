# helper function to visualize images
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf 
from matplotlib import pyplot as plt


def visualize(image):
    image = image.reshape(512, 512)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(image, cmap='bone')
    

    
def get_image(viewPosition, images):
    for i, vp in enumerate(viewPosition):
        if vp in [5, 12]:
            return np.concatenate((images[i], images[i], images[i]), axis=2)

def to_model_fit(sample):
    viewPosition = sample["viewPosition"]
    images = sample["image"]
    image = tf.py_function(get_image, [viewPosition, images], tf.uint16)
    labels = sample["label_chexpert"]
    return image, labels

def get_token():
    return {"aws_access_key_id": "",
            "aws_secret_access_key": "",    
            "aws_session_token": ""}


def iterate():
    for item in ds:
        print(item["label_chexpert"].compute()) # or you can access any other key from schema
        print(item["viewPosition"].compute()) # the ClassLabels are stored as integers
        print(item["viewPosition"].compute(label_name=True)) # strings labels are retrieved in this manner
        break
        
def get_model():
    
    inputs = tf.keras.Input([512, 512, 3])

    x = inputs
    x = tf.keras.layers.Conv2D(8, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = tf.keras.layers.Conv2D(8, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    c1 = x

    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    c2 = x

    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    c3 = x

    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    c4 = x

    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(14, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, x)
    model.summary()
    return model