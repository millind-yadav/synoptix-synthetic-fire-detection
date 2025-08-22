#!/usr/bin/env python3
"""
Main entry point for the Synoptix fire detection training pipeline.
Orchestrates the training process using GCP TPU/GPU/CPU.
"""

import argparse
from trainer import GCPTPUFireDetectionTrainer
from utils import install_packages, download_data_from_gcs

install_packages()

def main():
    """Parse arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(description='GCP TPU Fire Detection Training')
    parser.add_argument('--bucket', type=str, help='GCS bucket name for data and results')
    parser.add_argument('--experiments-path', type=str, default='data/ratio_experiments',
                        help='Local path to experiments')
    parser.add_argument('--results-path', type=str, default='results',
                        help='Local path for results')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of initial training epochs')
    parser.add_argument('--fine-tune-epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--download-data', action='store_true',
                        help='Download data from GCS bucket')
    
    args = parser.parse_args()
    
    print("🔥 Synoptix Fire Detection Training Pipeline")
    print("=" * 70)
    print("🚀 Google Cloud TPU/GPU/CPU Acceleration")
    print("🎯 Real vs Synthetic Data Ratio Analysis")
    print("=" * 70)
    
    # Download data if requested
    if args.download_data and args.bucket:
        experiments_path = download_data_from_gcs(args.bucket, args.experiments_path)
        if not experiments_path:
            print("❌ Failed to download data from GCS")
            return
    else:
        experiments_path = args.experiments_path
    
    # Initialize trainer
    trainer = GCPTPUFireDetectionTrainer(
        experiments_path=experiments_path,
        results_path=args.results_path,
        bucket_name=args.bucket
    )
    
    # Run training
    successful_results, failed_experiments = trainer.train_all_experiments(
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    if successful_results:
        print(f"\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"✅ {len(successful_results)} models trained and saved")
        print(f"📊 Comprehensive analysis generated")
        print(f"💾 All models ready for deployment")
    else:
        print(f"\n❌ TRAINING PIPELINE FAILED")
        print(f"💥 No models were successfully trained")

if __name__ == "__main__":
    main()
