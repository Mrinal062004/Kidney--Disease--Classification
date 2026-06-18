import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from Kidney.utils.common import decodeImage
from Kidney.pipeline.prediction import PredictionPipeline

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

        print("Loading model...")
        self.classifier = PredictionPipeline(
            "model/model.h5"
        )
        print("Model loaded successfully")


# IMPORTANT: Create object globally for Gunicorn
clApp = ClientApp()


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
    try:
        print("Predict route hit")

        image = request.json["image"]
        print("Image received")

        decodeImage(image, clApp.filename)
        print("Image decoded")

        result = clApp.classifier.predict()
        print("Prediction completed")

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port
    )