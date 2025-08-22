#!/usr/bin/env python3
"""
Single image inference script for fire detection using a trained ResNet50 model.
Predicts fire/no_fire on a given image and displays/saves the result.
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from metrics import MacroF1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'MacroF1': MacroF1},
            compile=False
        )
        logger.info(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from {model_path}: {e}")
        raise

def preprocess_image(image_path):
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        return image, image_input
    except Exception as e:
        logger.error(f"❌ Error preprocessing image {image_path}: {e}")
        raise

def get_prediction(model, image_input):
    try:
        prediction = model.predict(image_input, verbose=0)
        logger.info(f"Raw predictions: Fire={prediction[0][0]:.4f}, No Fire={prediction[0][1]:.4f}")
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        class_label = 'Fire' if class_idx == 0 else 'No Fire'
        return class_label, confidence
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Single Image Fire Detection Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--output-image', type=str, help='Path to save annotated image')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                        help='Confidence threshold for displaying predictions')
    
    args = parser.parse_args()

    model = load_model(args.model_path)
    original_image, image_input = preprocess_image(args.image_path)
    class_label, confidence = get_prediction(model, image_input)
    logger.info(f"Prediction: {class_label}, Confidence: {confidence:.2%}")

    if confidence >= args.confidence_threshold:
        text = f"{class_label}: {confidence:.2%}"
        color = (0, 0, 255) if class_label == 'Fire' else (0, 255, 0)
        cv2.putText(original_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    color, 2, cv2.LINE_AA)

    cv2.imshow('Fire Detection', original_image)
    logger.info("✅ Displaying image. Press any key to exit.")

    if args.output_image:
        try:
            output_path = Path(args.output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), original_image)
            logger.info(f"✅ Annotated image saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    logger.info("✅ Inference completed")

if __name__ == "__main__":
    main()