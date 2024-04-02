import logging
import scipy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

my_path = os.path.abspath(os.path.dirname(__file__))

inception_model_path = os.path.join(my_path, "../Models/Classification/resnet_pneumonia_detection.h5")
vgg16_model_path = os.path.join(my_path, "../Models/Classification/new_vgg16_pneumonia_detection (1).h5")
resnet_model_path = os.path.join(my_path, "../Models/Classification/Resnet50_pneumonia_detection.h5")


def predict_pneumonia(image):
    # Convert the input image to a list
    image = [image]

    # Load the models
    inception_model = load_model(inception_model_path)
    vgg16_model = load_model(vgg16_model_path)
    resnet_model = load_model(resnet_model_path)

    # List of models for ensemble
    models = [inception_model, vgg16_model, resnet_model]

    # Weights for ensemble based on model performance
    weights = [0.2, 0.3, 0.5]

    # Ensure that the sum of weights is 1
    weights = np.array(weights) / np.sum(weights)

    # DataFrame to hold the input image for prediction
    df_file = pd.DataFrame({'filename': image})

    # Data generator for the input image
    pred_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df_file,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    # Get predictions from each model
    preds = np.array([model.predict(pred_generator) for model in models])

    # Calculate the weighted average of predictions
    weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))

    binary_predictions = [1 if x > 0.6 else 0 for x in weighted_preds][0]

    # Map the prediction to the corresponding label
    label_map = {0: 'Normal', 1: 'Pneumonia'}
    final_prediction = label_map[binary_predictions]

    # Clean up models from memory
    del inception_model, vgg16_model, resnet_model

    return final_prediction


def predict_single_pneumonia(image):
    image = [image]

    df_file = pd.DataFrame({
        'filename': image
    })

    pred_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df_file,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    loaded_model = tf.keras.models.load_model(
        r'C:\Users\94703\PycharmProjects\lung-safe\Models\Classification\Resnet50_pneumonia_detection.h5')
    predictions = loaded_model.predict(pred_generator, steps=np.ceil(len(df_file) / 1))
    binary_predictions = [1 if x > 0.5 else 0 for x in predictions]
    print(binary_predictions)
