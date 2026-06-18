import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self):

        imagename = "inputImage.jpg"

        # Load image
        test_image = image.load_img(
            imagename,
            target_size=(128, 128)
        )

        # Convert to array
        test_image = image.img_to_array(test_image)

        # Same preprocessing as training
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(
            test_image,
            axis=0
        )

        # Predict
        pred = self.model.predict(test_image)

        print("=" * 50)
        print("Model Output Shape:", self.model.output_shape)
        print("Raw Prediction:", pred)
        print("=" * 50)

        # Binary classification model
        if self.model.output_shape[-1] == 1:

            probability = pred[0][0]

            print("Probability:", probability)

            if probability > 0.5:
                prediction = "Tumor"
            else:
                prediction = "Normal"

        # Multi-class model
        else:

            result = np.argmax(pred, axis=1)

            print("Predicted Class:", result)

            # CHANGE THIS IF YOUR CLASS INDICES ARE REVERSED
            if result[0] == 0:
                prediction = "Normal"
            else:
                prediction = "Tumor"

        print("Final Prediction:", prediction)

        return [{"image": prediction}]