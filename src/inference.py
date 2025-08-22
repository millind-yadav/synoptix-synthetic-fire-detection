#!/usr/bin/env python3
"""
Live video inference script for fire detection using a trained ResNet50 model.
Processes webcam or video stream, displays predictions, and optionally saves output.
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import yaml
import logging
from metrics import MacroF1  # Import custom metric

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the trained ResNet50 model with custom objects."""
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

def preprocess_frame(frame):
    """Preprocess video frame to match training pipeline (224x224, normalized)."""
    # Resize to 224x224
    frame_resized = cv2.resize(frame, (224, 224))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    frame_normalized = frame_rgb / 255.0
    # Add batch dimension
    frame_input = np.expand_dims(frame_normalized, axis=0)
    return frame_input

def get_prediction(model, frame):
    """Get fire/no_fire prediction with confidence."""
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame, verbose=0)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    class_label = 'Fire' if class_idx == 0 else 'No Fire'
    return class_label, confidence

def main():
    """Run live video inference with the trained model."""
    parser = argparse.ArgumentParser(description='Live Video Fire Detection Inference')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .h5 model')
    parser.add_argument('--video-source', type=str, default='0', 
                        help='Video source: "0" for webcam, or URL/path to video file')
    parser.add_argument('--output-video', type=str, help='Path to save output video')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                        help='Confidence threshold for displaying predictions')
    
    args = parser.parse_args()

    # Load configuration
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded configuration from config/config.yaml")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return

    # Load model
    model = load_model(args.model_path)

    # Initialize video capture
    try:
        if args.video_source.isdigit():
            cap = cv2.VideoCapture(int(args.video_source))
        else:
            cap = cv2.VideoCapture(args.video_source)
        if not cap.isOpened():
            raise ValueError("Failed to open video source")
        logger.info(f"✅ Video source opened: {args.video_source}")
    except Exception as e:
        logger.error(f"❌ Error opening video source {args.video_source}: {e}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Initialize video writer if output path provided
    out = None
    if args.output_video:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = Path(args.output_video)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            logger.info(f"✅ Video writer initialized: {output_path}")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize video writer: {e}")

    logger.info("🔥 Starting live video inference...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ End of video stream or failed to read frame")
                break

            # Get prediction
            class_label, confidence = get_prediction(model, frame)

            # Draw prediction on frame
            if confidence >= args.confidence_threshold:
                text = f"{class_label}: {confidence:.2%}"
                color = (0, 0, 255) if class_label == 'Fire' else (0, 255, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           color, 2, cv2.LINE_AA)

            # Display frame
            cv2.imshow('Fire Detection', frame)

            # Write frame to output video if enabled
            if out:
                out.write(frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("🛑 User stopped inference")
                break

    except Exception as e:
        logger.error(f"❌ Error during inference: {e}")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    main()