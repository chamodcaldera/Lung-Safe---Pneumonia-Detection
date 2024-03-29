from flask import Flask, request, jsonify, render_template
from Segmentation.Segmentation import seg_predict
from PIL import Image

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('nh.html')


@app.route("/predict", methods=["POST"])
def predict_seg():
    if "image" not in request.files:
        return "No image file in request", 400

    file = request.files["image"]
    image = Image.open(file.stream)  # Read the image in PIL format
    img_str = seg_predict(image)
    img_str = img_str.decode('utf-8')
    return render_template('nh.html', img_str=img_str)


if __name__ == '__main__':
    app.run()
