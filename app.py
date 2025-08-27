import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from keras import layers
import os

def cat_vs_dog_classification_model(input_shape=(128, 128, 3), num_classes=1):
    
    model = tf.keras.models.Sequential()

    
    model.add(layers.Rescaling(1./255, input_shape=input_shape))

    # BLOCK 1
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(strides=(2,2), pool_size=(2,2)))

    # BLOCK 2
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(strides=(2,2), pool_size=(2,2)))

    # BLOCK 3
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(strides=(2,2), pool_size=(2,2)))

    # BLOCK 4
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3,3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(strides=2, pool_size=(2,2)))

    # Fully connected layers
    model.add(layers.Flatten())

    model.add(layers.Dense(4096))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model



script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'cat_dog_model.weights.h5')


model = cat_vs_dog_classification_model()
try:
    model.load_weights(weights_path)
except FileNotFoundError:
    print(f"Error: Model weights file not found at '{weights_path}'. Please ensure it is in the same folder as this script.")
    model = None 



def classify_image(image: Image.Image):
    
    if model is None:
        return "Model not loaded. Please check the weights file.", 0.0

    
    image = image.resize((128, 128))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  

    
    prediction = model.predict(image)[0][0]

    
    if prediction > 0.5:
        return {"Dog": prediction, "Cat": 1 - prediction}
    else:        
        return {"Cat": 1 - prediction, "Dog": prediction}



iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image of a cat or a dog"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Cat vs. Dog Classifier",
    description="Upload an image and the model will predict if it is a cat or a dog.",
    live=True,
    theme="soft" 
)

if __name__ == "__main__":
    iface.launch(share=True)