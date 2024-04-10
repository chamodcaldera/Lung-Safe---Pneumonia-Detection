import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


def predict_pneumonia(image,inception_model,vgg16_model,resnet_model):
    image = [image]


    models = [inception_model, vgg16_model, resnet_model]

    # Weights for ensemble based on model performance
    weights = [0.2, 0.3, 0.5]


    weights = np.array(weights) / np.sum(weights)


    df_file = pd.DataFrame({'filename': image})


    pred_generator = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        df_file,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False
    )


    preds = np.array([model.predict(pred_generator) for model in models])


    weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))

    binary_predictions = [1 if x > 0.6 else 0 for x in weighted_preds][0]


    label_map = {0: 'Normal', 1: 'Pneumonia'}
    final_prediction = label_map[binary_predictions]


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
