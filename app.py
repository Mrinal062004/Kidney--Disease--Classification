import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from Kidney.utils.common import decodeImage
from Kidney.pipeline.prediction import PredictionPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)

# Model will be loaded only when first prediction is requested
clApp = None


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

        print("=" * 50)
        print("Loading model...")
        print("=" * 50)

        self.classifier = PredictionPipeline(
            "model/model.h5"
        )

        print("=" * 50)
        print("Model loaded successfully")
        print("=" * 50)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training Successfully Done"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    global clApp

    try:
        print("=" * 50)
        print("Predict route hit")
        print("=" * 50)

        # Load model only once
        if clApp is None:
            print("Loading model for first prediction...")
            clApp = ClientApp()

        image_data = request.json["image"]

        print("Image received")

        decodeImage(
            image_data,
            clApp.filename
        )

        print("Image decoded and saved")

        result = clApp.classifier.predict()

        print("Prediction completed")
        print(result)

        return jsonify(result)

    except Exception as e:

        print("=" * 50)
        print("ERROR OCCURRED")
        print(str(e))
        print("=" * 50)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )