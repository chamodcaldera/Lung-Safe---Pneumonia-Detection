from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64


def IoU(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def DiceScore(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice


def Precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    true_positives = tf.reduce_sum(y_true * y_pred)
    predicted_positives = tf.reduce_sum(y_pred)
    precision = true_positives / (predicted_positives + 1e-6)
    return precision


def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize(target_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img



model_path = 'v_unet_model.h5'
custom_objects = {'IoU': IoU, 'DiceScore': DiceScore, 'Precision': Precision}
model = load_model(model_path, custom_objects=custom_objects)


def seg_predict(image):
    img = preprocess_image(image, (256, 256))
    pred_mask = model.predict(img)
    threshold = 0.5
    pred_mask = (pred_mask > threshold).astype(np.uint8)

    pred_mask = pred_mask.squeeze() * 255
    pred_mask_image = Image.fromarray(pred_mask)

    buffer = BytesIO()
    pred_mask_image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    return img_str
