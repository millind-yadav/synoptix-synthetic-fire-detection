#!/usr/bin/env python3
"""
Utility functions for the Synoptix fire detection project.
Includes package installation, GCS operations, TPU setup, and plotting.
"""

import os
import subprocess
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import mixed_precision
from google.cloud import storage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def install_packages():
    """Install required packages."""
    packages = [
        'pandas>=1.3.0', 'numpy>=1.21.0', 'matplotlib>=3.5.0', 'seaborn>=0.11.0',
        'scikit-learn>=1.0.0', 'google-cloud-storage>=2.0.0', 'Pillow>=8.0.0'
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"⚠️ Warning installing {package}: {e}")

def download_data_from_gcs(bucket_name, local_path="data"):
    """Download training data from Google Cloud Storage."""
    print(f"📥 Downloading data from gs://{bucket_name} to {local_path}/...")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        Path(local_path).mkdir(exist_ok=True)
        blobs = list(bucket.list_blobs(prefix='ratio_experiments/'))
        if not blobs:
            print("❌ No files found with prefix 'ratio_experiments/'")
            return None
        print(f"🔍 Found {len(blobs)} files to download")
        downloaded_files = 0
        for blob in blobs:
            if not blob.name.endswith('/'):
                local_file_path = Path(local_path) / blob.name
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_file_path))
                downloaded_files += 1
                if downloaded_files % 100 == 0:
                    print(f"📥 Downloaded {downloaded_files} files...")
        print(f"✅ Downloaded {downloaded_files} files successfully")
        return str(Path(local_path) / "ratio_experiments")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

def upload_results_to_gcs(bucket_name, local_results_path, gcs_results_path="training_results"):
    """Upload training results to Google Cloud Storage."""
    print(f"📤 Uploading results to gs://{bucket_name}/{gcs_results_path}/...")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        uploaded_files = 0
        for local_file in Path(local_results_path).rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_results_path)
                gcs_path = f"{gcs_results_path}/{relative_path}"
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_file))
                uploaded_files += 1
                if uploaded_files % 50 == 0:
                    print(f"📤 Uploaded {uploaded_files} files...")
        print(f"✅ Uploaded {uploaded_files} files to gs://{bucket_name}/{gcs_results_path}/")
        return True
    except Exception as e:
        print(f"❌ Error uploading results: {e}")
        return False

def setup_gcp_tpu():
    """Setup TPU for Google Cloud training."""
    try:
        print("🔧 Initializing TPU configuration...")
        tpu_name = os.environ.get("TPU_NAME", "local")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("🚀 GCP TPU initialized successfully!")
        print(f"📊 Number of TPU cores: {strategy.num_replicas_in_sync}")
        return strategy, True, f"TPU({tpu_name})_x{strategy.num_replicas_in_sync}"
    except Exception as e:
        print(f"⚠️ TPU initialization failed: {e}")
        print("🔧 Attempting GPU fallback...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategy = tf.distribute.MirroredStrategy()
                print("✅ GPU acceleration enabled")
                return strategy, False, f"GPU_x{len(gpus)}"
            except Exception as gpu_error:
                print(f"❌ GPU setup failed: {gpu_error}")
        print("🔧 Using CPU (no GPU detected)")
        return tf.distribute.get_strategy(), False, "CPU"

def plot_training_history(history, exp_name, save_dir):
    """Create training history visualizations."""
    print(f"📊 Generating training history plots for {exp_name}...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'GCP TPU Training History - {exp_name}', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2, color='blue')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'precision' in history and 'val_precision' in history:
        axes[0, 2].plot(history['precision'], label='Training Precision', linewidth=2, color='blue')
        axes[0, 2].plot(history['val_precision'], label='Validation Precision', linewidth=2, color='orange')
        axes[0, 2].set_title('Model Precision', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)
    
    if 'recall' in history and 'val_recall' in history:
        axes[1, 0].plot(history['recall'], label='Training Recall', linewidth=2, color='blue')
        axes[1, 0].plot(history['val_recall'], label='Validation Recall', linewidth=2, color='orange')
        axes[1, 0].set_title('Model Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
    
    if 'lr' in history or 'learning_rate' in history:
        lr_key = 'lr' if 'lr' in history else 'learning_rate'
        axes[1, 1].plot(history[lr_key], label='Learning Rate', linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    epochs = range(1, len(history['accuracy']) + 1)
    axes[1, 2].plot(epochs, history['accuracy'], 'b-', label='Training Acc', linewidth=2)
    axes[1, 2].plot(epochs, history['val_accuracy'], 'r-', label='Validation Acc', linewidth=2)
    axes[1, 2].set_title('Training Progress Summary', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = save_dir / f"{exp_name}_training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Training history plot saved: {plot_path}")
    return str(plot_path)

def plot_confusion_matrix(confusion_matrix_data, class_names, exp_name, save_dir):
    """Create confusion matrix visualizations."""
    print(f"📊 Generating confusion matrix for {exp_name}...")
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Confusion Matrix Analysis - {exp_name}', fontsize=14, fontweight='bold')
    
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Raw Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    cm_normalized = confusion_matrix_data.astype('float') / confusion_matrix_data.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                cbar_kws={'label': 'Percentage'})
    ax2.set_title('Normalized Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plot_path = save_dir / f"{exp_name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Confusion matrix plot saved: {plot_path}")
    return str(plot_path)
