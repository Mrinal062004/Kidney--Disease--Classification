import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, model_path):
        print("Loading model from:", model_path)

        self.model = load_model(model_path)

        print("Model loaded successfully")
        print("Model output shape:", self.model.output_shape)

    def predict(self):

        imagename = "inputImage.jpg"

        print("=" * 50)
        print("Starting prediction")
        print("Image path:", imagename)
        print("File exists:", os.path.exists(imagename))
        print("=" * 50)

        if not os.path.exists(imagename):
            return [{"error": f"{imagename} not found"}]

        try:
            # Load image
            test_image = image.load_img(
                imagename,
                target_size=(128, 128)
            )

            print("Image loaded successfully")

            # Convert image to array
            test_image = image.img_to_array(test_image)

            print("Converted to array")
            print("Shape:", test_image.shape)

            # Normalize
            test_image = test_image / 255.0

            # Add batch dimension
            test_image = np.expand_dims(
                test_image,
                axis=0
            )

            print("Final input shape:", test_image.shape)

            print("Running model prediction...")

            pred = self.model.predict(
                test_image,
                verbose=1
            )

            print("Prediction completed")

            print("=" * 50)
            print("Raw Prediction:", pred)
            print("=" * 50)

            # Binary classification
            if self.model.output_shape[-1] == 1:

                probability = float(pred[0][0])

                print("Probability:", probability)

                if probability > 0.5:
                    prediction = "Tumor"
                else:
                    prediction = "Normal"

            # Multi-class classification
            else:

                result = np.argmax(pred, axis=1)

                print("Predicted Class Index:", result)

                if result[0] == 0:
                    prediction = "Normal"
                else:
                    prediction = "Tumor"

            print("Final Prediction:", prediction)

            return [{"image": prediction}]

        except Exception as e:

            print("ERROR DURING PREDICTION:")
            print(str(e))

            return [{"error": str(e)}]