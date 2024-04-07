import base64
import tempfile
import io
import os
from flask import Flask, request, jsonify, render_template
from Functions.Segmentation import seg_predict
from Functions.Classification import predict_pneumonia , predict_single_pneumonia
from PIL import Image

app = Flask(__name__)


def save_base64_image_to_temp(base64_str):
    # Decode the base64 string
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))


    static_dir = os.path.join(app.root_path, 'static')
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=static_dir, suffix='.png')
    temp_file_path = temp_file.name
    image.save(temp_file_path, 'PNG')

    return temp_file_path
@app.route('/')
def index():
    return render_template('classify.html')


@app.route("/predict", methods=["POST"])
def predict_seg():
    if "image" not in request.files:
        return "No image file in request", 400

    file = request.files["image"]

    if file.content_type not in ['image/jpeg', 'image/png']:
        return "Unsupported image format. Please upload JPEG or PNG images only.", 400
    image = Image.open(file.stream)  # Read the image in PIL format
    img_str = seg_predict(image)
    img_str = img_str.decode('utf-8')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    return render_template('segmentation.html', img_str=img_str,img_data=encoded_img)


@app.route("/classify", methods=["POST"])
def detect_pneumonia():
    if "image" not in request.files:
        return "No image file in request", 400

    file = request.files["image"]


    if file.content_type not in ['image/jpeg', 'image/png']:
        return "Unsupported image format. Please upload JPEG or PNG images only.", 400


    image = Image.open(file.stream)  # Read the image in PIL format
    img_str = seg_predict(image)
    temp_image_path = save_base64_image_to_temp(img_str)
    img_str = img_str.decode('utf-8')

    category= predict_pneumonia(temp_image_path)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    os.remove(temp_image_path)

    return render_template('classify.html', category=category,img_str=img_str,img_data=encoded_img)


if __name__ == '__main__':
    app.run()
