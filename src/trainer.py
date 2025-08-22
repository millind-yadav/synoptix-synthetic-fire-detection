#!/usr/bin/env python3
"""
Core trainer class for the Synoptix fire detection project.
Handles model training, evaluation, and result generation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import yaml
from .utils import setup_gcp_tpu, plot_training_history, plot_confusion_matrix
from .metrics import MacroF1

with open('config/config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

class GCPTPUFireDetectionTrainer:
    """Trainer for fire detection using ResNet50 on GCP TPU/GPU/CPU."""
    def __init__(self, experiments_path=CONFIG['paths']['experiments'], results_path=CONFIG['paths']['results'], bucket_name=None):
        self.bucket_name = bucket_name
        self.experiments_path = Path(experiments_path)
        self.results_path = Path(results_path)
        
        self.strategy, self.using_tpu, self.device_info = setup_gcp_tpu()
        
        if self.using_tpu:
            mixed_precision.set_global_policy('mixed_bfloat16')
            self.batch_size = CONFIG['training']['batch_size_tpu']
            print("⚡ Using mixed_bfloat16 on TPU")
        elif "GPU" in self.device_info:
            self.batch_size = CONFIG['training']['batch_size_gpu']
        else:
            self.batch_size = CONFIG['training']['batch_size_cpu']
            
        print(f"🎯 Using batch size: {self.batch_size} (optimized for {self.device_info})")
        
        self.setup_results_directories()
        self.find_experiments()
    
    def compute_class_weight_from_dirs(self, train_dir: Path):
        """Compute class weights from directory counts."""
        exts = ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG','*.bmp','*.BMP']
        def count_many(p: Path):
            return sum(len(list(p.glob(e))) for e in exts)
    
        n_fire = count_many(train_dir / 'fire')
        n_nofire = count_many(train_dir / 'no_fire')
        total = n_fire + n_nofire
        w_fire = total / (2.0 * max(1, n_fire))
        w_nofire = total / (2.0 * max(1, n_nofire))
        return {0: w_fire, 1: w_nofire}, (n_fire, n_nofire)
    
    def setup_results_directories(self):
        """Create directory structure for results."""
        directories = [
            self.results_path,
            self.results_path / "models",
            self.results_path / "plots" / "training_history",
            self.results_path / "plots" / "confusion_matrices", 
            self.results_path / "plots" / "class_performance",
            self.results_path / "plots" / "comparison_analysis",
            self.results_path / "logs",
            self.results_path / "experiment_details",
            self.results_path / "csv_exports"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"📁 Results directory structure created: {self.results_path}")
    
    def find_experiments(self):
        """Find and validate experiment directories."""
        self.experiments = []
        if not self.experiments_path.exists():
            print(f"❌ Experiments path not found: {self.experiments_path}")
            return
        for d in self.experiments_path.iterdir():
            if d.is_dir() and d.name.endswith('S'):
                required_dirs = ["train/fire", "train/no_fire", "val/fire", "val/no_fire", "test/fire", "test/no_fire"]
                valid = True
                for req_dir in required_dirs:
                    if not (d / req_dir).exists():
                        print(f"⚠️ Invalid experiment {d.name}: missing {req_dir}")
                        valid = False
                        break
                if valid:
                    self.experiments.append(d)
                else:
                    print(f"❌ Skipping invalid experiment: {d.name}")
        self.experiments.sort(key=lambda x: int(x.name.split('R')[0]), reverse=True)
        print(f"🔍 Found {len(self.experiments)} valid experiments:")
        for exp in self.experiments:
            print(f"   ✅ {exp.name}")
    
    def build_fire_detection_model(self):
        """Build ResNet50 model for fire detection."""
        with self.strategy.scope():
            base_model = ResNet50(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet',
                name='resnet50_backbone'
            )
            base_model.trainable = False
            inputs = tf.keras.Input(shape=(224, 224, 3), name='input_images')
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D(name='global_avg_pool')(x)
            x = Dropout(0.30, name='dropout_1')(x)
            x = Dense(256, activation='relu', name='feature_dense')(x)
            x = Dropout(0.20, name='dropout_2')(x)
            outputs = Dense(2, activation='softmax', name='fire_classifier', dtype='float32')(x)
            model = Model(inputs, outputs, name='ResNet50_FireDetection')
            base_lr = 3e-4
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
            model.compile(
                optimizer=Adam(learning_rate=base_lr, beta_1=0.9, beta_2=0.999),
                loss=loss,
                metrics=[CategoricalAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), MacroF1(name='macro_f1')]
            )
            print(f"🧠 ResNet50 Fire Detection Model Built:")
            print(f"   📊 Learning rate: {base_lr}")
            print(f"   🎯 Total parameters: {model.count_params():,}")
            print(f"   🔧 Device optimization: {self.device_info}")
            return model

    def create_tpu_optimized_datasets(self, exp_dir):
        """Create optimized tf.data datasets."""
        def preprocess_image(image_path, label):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, [224, 224])
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        
        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_hue(image, 0.1)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.image.rot90(image, tf.random.uniform([], 0, 4, dtype=tf.int32))
            return image, label
        
        def create_dataset_from_directory(directory, augment=False):
            fire_dir = directory / "fire"
            no_fire_dir = directory / "no_fire"
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']
            fire_paths = []
            no_fire_paths = []
            for ext in extensions:
                fire_paths.extend(list(fire_dir.glob(ext)))
                no_fire_paths.extend(list(no_fire_dir.glob(ext)))
            print(f"   📂 {directory.name}: Fire={len(fire_paths)}, No Fire={len(no_fire_paths)}")
            if len(fire_paths) == 0 or len(no_fire_paths) == 0:
                raise ValueError(f"Insufficient data in {directory}: Fire={len(fire_paths)}, No Fire={len(no_fire_paths)}")
            all_paths = [str(p) for p in fire_paths] + [str(p) for p in no_fire_paths]
            all_labels = [0] * len(fire_paths) + [1] * len(no_fire_paths)
            all_labels_categorical = tf.keras.utils.to_categorical(all_labels, 2)
            dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels_categorical))
            dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
            if augment:
                dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            return dataset, len(all_paths)
        
        print(f"📂 Creating datasets for {exp_dir.name}...")
        train_dataset, train_count = create_dataset_from_directory(exp_dir / "train", augment=True)
        val_dataset, val_count = create_dataset_from_directory(exp_dir / "val", augment=False)
        test_dataset, test_count = create_dataset_from_directory(exp_dir / "test", augment=False)
        drop_remainder = True if self.using_tpu else False
        train_dataset = (train_dataset
                        .shuffle(buffer_size=min(1000, train_count), reshuffle_each_iteration=True)
                        .batch(self.batch_size, drop_remainder=drop_remainder)
                        .prefetch(tf.data.AUTOTUNE))
        val_dataset = (val_dataset
                      .batch(self.batch_size, drop_remainder=drop_remainder)
                      .prefetch(tf.data.AUTOTUNE))
        test_dataset = (test_dataset
                       .batch(self.batch_size, drop_remainder=drop_remainder)
                       .prefetch(tf.data.AUTOTUNE))
        steps_per_epoch = train_count // self.batch_size
        validation_steps = val_count // self.batch_size
        print(f"✅ Datasets created successfully:")
        print(f"   📊 Training: {train_count} images, {steps_per_epoch} steps/epoch")
        print(f"   📊 Validation: {val_count} images, {validation_steps} steps")
        print(f"   📊 Testing: {test_count} images")
        print(f"   🎯 Batch size: {self.batch_size}")
        return train_dataset, val_dataset, test_dataset, train_count, val_count, test_count

    def fine_tune_model(self, model, train_dataset, val_dataset, class_weight, fine_tune_epochs=10):
        """Fine-tune the ResNet50 model."""
        print("🔧 Starting fine-tuning phase...")
        with self.strategy.scope():
            base_model = model.get_layer('resnet50_backbone')
            base_model.trainable = True
            fine_tune_at = 140
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for l in base_model.layers[fine_tune_at:]:
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = False
            fine_tune_lr = 1e-5
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
            model.compile(
                optimizer=Adam(learning_rate=fine_tune_lr),
                loss=loss,
                metrics=[CategoricalAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), MacroF1(name='macro_f1')]
            )
            print(f"📊 Fine-tuning learning rate: {fine_tune_lr}")
            print(f"🎯 Unfroze ~{len(base_model.layers) - fine_tune_at} layers (BNs kept frozen)")
        fine_tune_history = model.fit(
            train_dataset,
            epochs=fine_tune_epochs,
            validation_data=val_dataset,
            class_weight=class_weight,
            verbose=1
        )
        print(f"✅ Fine-tuning completed ({fine_tune_epochs} epochs)")
        return fine_tune_history

    def evaluate_model_comprehensive(self, model, test_dataset, test_count, exp_name):
        """Evaluate model with detailed metrics."""
        print(f"📊 Performing comprehensive evaluation for {exp_name}...")
        predictions = model.predict(test_dataset, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = []
        for batch_data, batch_labels in test_dataset:
            y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_true = np.array(y_true)
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        predictions = predictions[:min_len]
        print(f"   📊 Evaluated on {min_len} samples")
        test_accuracy = np.mean(y_true == y_pred)
        class_names = ['fire', 'no_fire']
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        confidences = np.max(predictions, axis=1)
        avg_confidence = np.mean(confidences)
        evaluation_results = {
            'test_accuracy': float(test_accuracy),
            'avg_confidence': float(avg_confidence),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'samples_evaluated': int(min_len)
        }
        print(f"   ✅ Evaluation completed - Accuracy: {test_accuracy:.4f}")
        return evaluation_results, y_true, y_pred, predictions, cm

    def train_single_experiment(self, exp_dir, epochs=CONFIG['training']['epochs'], fine_tune_epochs=CONFIG['training']['fine_tune_epochs']):
        """Train a single experiment."""
        exp_name = exp_dir.name
        print(f"\n{'='*70}")
        print(f"🚀 Training GCP TPU ResNet50: {exp_name}")
        print(f"🔥 Device: {self.device_info}")
        print(f"🎯 TPU Acceleration: {self.using_tpu}")
        print(f"{'='*70}")
        
        metadata_path = exp_dir / "experiment_metadata.json"
        metadata = {"experiment_name": exp_name}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata.update(json.load(f))
                    print(f"📋 Loaded experiment metadata:")
                    print(f"   Real Data: {metadata.get('real_percentage', 'Unknown')}%")
                    print(f"   Synthetic Data: {metadata.get('synthetic_percentage', 'Unknown')}%")
                    print(f"   Total Images: {metadata.get('total_images', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
        
        training_start_time = datetime.now()
        try:
            train_dataset, val_dataset, test_dataset, train_count, val_count, test_count = self.create_tpu_optimized_datasets(exp_dir)
            print(f"🧠 Building ResNet50 model...")
            model = self.build_fire_detection_model()
            model_save_path = self.results_path / "models" / f"{exp_name}_best_model.h5"
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, mode='min', verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7, mode='min', verbose=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=str(model_save_path), monitor='val_loss', mode='min', save_best_only=True, verbose=1),
                tf.keras.callbacks.CSVLogger(filename=str(self.results_path / "logs" / f"{exp_name}_training_log.csv"), append=False)
            ]
            print(f"🚀 Starting initial training ({epochs} epochs)...")
            initial_start = datetime.now()
            class_weight, (n_fire, n_nofire) = self.compute_class_weight_from_dirs(exp_dir / "train")
            print(f"⚖️ Class weights -> fire: {class_weight[0]:.2f}, no_fire: {class_weight[1]:.2f} "
                  f"(counts: fire={n_fire}, no_fire={n_nofire})")
            history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, class_weight=class_weight, verbose=1)
            initial_time = (datetime.now() - initial_start).total_seconds() / 60
            actual_epochs = len(history.history['loss'])
            print(f"✅ Initial training completed in {initial_time:.1f} minutes ({actual_epochs} epochs)")
            initial_model_path = self.results_path / "models" / f"{exp_name}_initial_model.h5"
            model.save(initial_model_path)
            print(f"🔧 Starting fine-tuning ({fine_tune_epochs} epochs)...")
            fine_tune_start = datetime.now()
            fine_tune_history = self.fine_tune_model(model, train_dataset, val_dataset, class_weight, fine_tune_epochs)
            fine_tune_time = (datetime.now() - fine_tune_start).total_seconds() / 60
            print(f"✅ Fine-tuning completed in {fine_tune_time:.1f} minutes")
            final_model_path = self.results_path / "models" / f"{exp_name}_final_model.h5"
            model.save(final_model_path)
            total_training_time = initial_time + fine_tune_time
            total_epochs = actual_epochs + len(fine_tune_history.history['loss'])
            combined_history = {}
            all_keys = set(history.history.keys()) | set(fine_tune_history.history.keys())
            for key in all_keys:
                hist1 = history.history.get(key, [])
                hist2 = fine_tune_history.history.get(key, [])
                combined_history[key] = hist1 + hist2
            eval_results, y_true, y_pred, predictions, cm = self.evaluate_model_comprehensive(model, test_dataset, test_count, exp_name)
            print(f"📊 Generating comprehensive visualizations...")
            plot_paths = {
                'training_history': plot_training_history(combined_history, exp_name, self.results_path / "plots" / "training_history"),
                'confusion_matrix': plot_confusion_matrix(cm, ['Fire', 'No Fire'], exp_name, self.results_path / "plots" / "confusion_matrices")
            }
            experiment_result = {
                "experiment_info": {
                    "name": exp_name,
                    "real_percentage": metadata.get("real_percentage", 0),
                    "synthetic_percentage": metadata.get("synthetic_percentage", 0),
                    "total_images": metadata.get("total_images", 0),
                    "training_device": self.device_info,
                    "tpu_accelerated": self.using_tpu
                },
                "dataset_info": {
                    "training_samples": train_count,
                    "validation_samples": val_count,
                    "test_samples": test_count,
                    "evaluated_samples": eval_results['samples_evaluated'],
                    "batch_size": self.batch_size,
                    "classes": ['fire', 'no_fire']
                },
                "training_info": {
                    "initial_epochs_planned": epochs,
                    "initial_epochs_completed": actual_epochs,
                    "fine_tune_epochs": len(fine_tune_history.history['loss']),
                    "total_epochs": total_epochs,
                    "initial_training_time_minutes": initial_time,
                    "fine_tune_time_minutes": fine_tune_time,
                    "total_training_time_minutes": total_training_time,
                    "early_stopping_triggered": actual_epochs < epochs,
                    "training_completed_at": datetime.now().isoformat()
                },
                "performance_metrics": {
                    "test_accuracy": eval_results['test_accuracy'],
                    "avg_prediction_confidence": eval_results['avg_confidence'],
                    "best_val_accuracy": float(max(combined_history['val_accuracy'])),
                    "final_val_accuracy": float(combined_history['val_accuracy'][-1])
                },
                "detailed_metrics": {
                    "fire_precision": float(eval_results['classification_report']['fire']['precision']),
                    "fire_recall": float(eval_results['classification_report']['fire']['recall']),
                    "fire_f1_score": float(eval_results['classification_report']['fire']['f1-score']),
                    "no_fire_precision": float(eval_results['classification_report']['no_fire']['precision']),
                    "no_fire_recall": float(eval_results['classification_report']['no_fire']['recall']),
                    "no_fire_f1_score": float(eval_results['classification_report']['no_fire']['f1-score']),
                    "balanced_f1_score": (eval_results['classification_report']['fire']['f1-score'] + 
                                         eval_results['classification_report']['no_fire']['f1-score']) / 2
                },
                "confusion_matrix": eval_results['confusion_matrix'],
                "model_paths": {
                    "initial_model": str(initial_model_path),
                    "final_model": str(final_model_path),
                    "best_model": str(model_save_path),
                    "training_log": str(self.results_path / "logs" / f"{exp_name}_training_log.csv")
                },
                "visualization_paths": plot_paths,
                "training_history": {
                    k: [float(x) for x in v] for k, v in combined_history.items() 
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0
                },
                "full_classification_report": eval_results['classification_report']
            }
            results_file = self.results_path / "experiment_details" / f"{exp_name}_complete_results.json"
            with open(results_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
            print(f"💾 Comprehensive results saved: {results_file}")
            print(f"\n✅ {exp_name} TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*50}")
            print(f"🎯 Device: {self.device_info}")
            print(f"📈 Test Accuracy: {eval_results['test_accuracy']:.4f}")
            print(f"🔥 Fire Detection F1: {experiment_result['detailed_metrics']['fire_f1_score']:.4f}")
            print(f"🚫 No Fire Detection F1: {experiment_result['detailed_metrics']['no_fire_f1_score']:.4f}")
            print(f"⚖️ Balanced F1 Score: {experiment_result['detailed_metrics']['balanced_f1_score']:.4f}")
            print(f"⏱️ Total Training Time: {total_training_time:.1f} minutes")
            print(f"🎯 Total Epochs: {total_epochs} (Initial: {actual_epochs}, Fine-tune: {len(fine_tune_history.history['loss'])})")
            print(f"💾 Models Saved: 3 versions (initial, final, best)")
            print(f"{'='*50}")
            return experiment_result
        except Exception as e:
            training_time = (datetime.now() - training_start_time).total_seconds() / 60
            error_message = f"Training failed for {exp_name} after {training_time:.1f} minutes: {str(e)}"
            print(f"❌ {error_message}")
            error_result = {
                "experiment_name": exp_name,
                "device": self.device_info,
                "status": "failed",
                "error": error_message,
                "error_type": type(e).__name__,
                "training_time_minutes": training_time,
                "timestamp": datetime.now().isoformat()
            }
            error_file = self.results_path / "experiment_details" / f"{exp_name}_error.json"
            with open(error_file, 'w') as f:
                json.dump(error_result, f, indent=2)
            return None

    def train_all_experiments(self, epochs=CONFIG['training']['epochs'], fine_tune_epochs=CONFIG['training']['fine_tune_epochs']):
        """Train all ratio experiments."""
        from .utils import upload_results_to_gcs
        print(f"🔥 GCP TPU FIRE DETECTION TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"🚀 Device: {self.device_info}")
        print(f"🎯 Training {len(self.experiments)} ratio experiments")
        print(f"📁 Results will be saved to: {self.results_path}")
        print(f"⚙️ Configuration: {epochs} initial epochs + {fine_tune_epochs} fine-tune epochs")
        print(f"🔢 Batch Size: {self.batch_size}")
        print(f"{'='*70}")

        successful_results = []
        failed_experiments = []
        pipeline_start_time = datetime.now()

        for i, exp_dir in enumerate(self.experiments, 1):
            print(f"\n🔄 Pipeline Progress: {i}/{len(self.experiments)} experiments")
            elapsed_time = (datetime.now() - pipeline_start_time).total_seconds() / 60
            print(f"⏰ Elapsed time: {elapsed_time:.1f} minutes")
            result = self.train_single_experiment(exp_dir, epochs, fine_tune_epochs)
            if result:
                successful_results.append(result)
                acc = result['performance_metrics']['test_accuracy']
                training_time = result['training_info']['total_training_time_minutes']
                print(f"✅ {exp_dir.name} completed successfully (Accuracy: {acc:.4f}, Time: {training_time:.1f}m)")
            else:
                failed_experiments.append(exp_dir.name)
                print(f"❌ {exp_dir.name} failed")

        total_pipeline_time = (datetime.now() - pipeline_start_time).total_seconds() / 60
        print(f"\n🎉 TRAINING PIPELINE COMPLETED!")
        print(f"🚀 Device: {self.device_info}")
        print(f"⏰ Total pipeline time: {total_pipeline_time:.1f} minutes")
        print(f"✅ Successful experiments: {len(successful_results)}")
        print(f"❌ Failed experiments: {len(failed_experiments)}")
        if failed_experiments:
            print(f"💥 Failed: {', '.join(failed_experiments)}")
        if successful_results:
            self.generate_comprehensive_pipeline_analysis(successful_results, failed_experiments, total_pipeline_time)
        return successful_results, failed_experiments

    def generate_comprehensive_pipeline_analysis(self, successful_results, failed_experiments, total_pipeline_time):
        """Generate comprehensive analysis across experiments."""
        print(f"\n📊 GENERATING COMPREHENSIVE PIPELINE ANALYSIS")
        print(f"{'='*70}")
        from .utils import upload_results_to_gcs
        if not successful_results:
            print("❌ No successful results to analyze")
            return
        summary_data = []
        for result in successful_results:
            exp_info = result['experiment_info']
            performance = result['performance_metrics']
            detailed = result['detailed_metrics']
            training_info = result['training_info']
            summary_data.append({
                'experiment': exp_info['name'],
                'real_percentage': exp_info['real_percentage'],
                'synthetic_percentage': exp_info['synthetic_percentage'],
                'test_accuracy': performance['test_accuracy'],
                'fire_f1_score': detailed['fire_f1_score'],
                'no_fire_f1_score': detailed['no_fire_f1_score'],
                'balanced_f1_score': detailed['balanced_f1_score'],
                'avg_confidence': performance['avg_prediction_confidence'],
                'training_time_minutes': training_info['total_training_time_minutes'],
                'total_epochs': training_info['total_epochs'],
                'device': exp_info['training_device'],
                'tpu_accelerated': exp_info['tpu_accelerated'],
                'model_path': result['model_paths']['final_model']
            })
        df = pd.DataFrame(summary_data)
        df_by_accuracy = df.sort_values('test_accuracy', ascending=False)
        df_by_balanced_f1 = df.sort_values('balanced_f1_score', ascending=False)
        df_by_fire_f1 = df.sort_values('fire_f1_score', ascending=False)
        csv_dir = self.results_path / "csv_exports"
        df_by_accuracy.to_csv(csv_dir / "results_by_accuracy.csv", index=False)
        df_by_balanced_f1.to_csv(csv_dir / "results_by_balanced_f1.csv", index=False)
        df_by_fire_f1.to_csv(csv_dir / "results_by_fire_detection.csv", index=False)
        df.to_csv(csv_dir / "complete_results_summary.csv", index=False)
        best_accuracy = df_by_accuracy.iloc[0]
        best_fire_detection = df_by_fire_f1.iloc[0]
        best_balanced = df_by_balanced_f1.iloc[0]
        performance_stats = {
            "accuracy_range": [float(df['test_accuracy'].min()), float(df['test_accuracy'].max())],
            "accuracy_mean": float(df['test_accuracy'].mean()),
            "accuracy_std": float(df['test_accuracy'].std()),
            "fire_f1_range": [float(df['fire_f1_score'].min()), float(df['fire_f1_score'].max())],
            "fire_f1_mean": float(df['fire_f1_score'].mean()),
            "balanced_f1_range": [float(df['balanced_f1_score'].min()), float(df['balanced_f1_score'].max())],
            "balanced_f1_mean": float(df['balanced_f1_score'].mean()),
            "avg_training_time_minutes": float(df['training_time_minutes'].mean()),
            "total_training_time_minutes": float(df['training_time_minutes'].sum()),
            "device_type": self.device_info,
            "tpu_acceleration": self.using_tpu
        }
        final_summary = {
            "pipeline_metadata": {
                "analysis_generated_at": datetime.now().isoformat(),
                "total_experiments_attempted": len(successful_results) + len(failed_experiments),
                "successful_experiments": len(successful_results),
                "failed_experiments": len(failed_experiments),
                "total_pipeline_time_minutes": total_pipeline_time,
                "device_type": self.device_info,
                "tpu_accelerated": self.using_tpu,
                "batch_size": self.batch_size
            },
            "performance_statistics": performance_stats,
            "best_performers": {
                "highest_accuracy": {
                    "experiment": best_accuracy['experiment'],
                    "accuracy": float(best_accuracy['test_accuracy']),
                    "balanced_f1": float(best_accuracy['balanced_f1_score']),
                    "training_time_minutes": float(best_accuracy['training_time_minutes']),
                    "data_composition": f"{best_accuracy['real_percentage']}% Real / {best_accuracy['synthetic_percentage']}% Synthetic",
                    "model_path": best_accuracy['model_path']
                },
                "best_fire_detection": {
                    "experiment": best_fire_detection['experiment'],
                    "fire_f1_score": float(best_fire_detection['fire_f1_score']),
                    "accuracy": float(best_fire_detection['test_accuracy']),
                    "data_composition": f"{best_fire_detection['real_percentage']}% Real / {best_fire_detection['synthetic_percentage']}% Synthetic",
                    "model_path": best_fire_detection['model_path']
                },
                "most_balanced": {
                    "experiment": best_balanced['experiment'],
                    "balanced_f1_score": float(best_balanced['balanced_f1_score']),
                    "accuracy": float(best_balanced['test_accuracy']),
                    "data_composition": f"{best_balanced['real_percentage']}% Real / {best_balanced['synthetic_percentage']}% Synthetic",
                    "model_path": best_balanced['model_path']
                }
            },
            "failed_experiments": failed_experiments,
            "complete_experiment_results": df_by_accuracy.to_dict('records')
        }
        final_summary_file = self.results_path / "COMPREHENSIVE_PIPELINE_ANALYSIS.json"
        with open(final_summary_file, 'w') as f:
            json.dump(final_summary, f, indent=2)
        print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"{'='*70}")
        print(f"🚀 Device: {self.device_info}")
        print(f"⏰ Total Pipeline Time: {total_pipeline_time:.1f} minutes")
        print(f"✅ Successful Experiments: {len(successful_results)}")
        print(f"❌ Failed Experiments: {len(failed_experiments)}")
        print(f"\n🏆 TOP PERFORMERS:")
        print(f"   🥇 Best Accuracy: {best_accuracy['experiment']} ({best_accuracy['test_accuracy']:.4f})")
        print(f"   🔥 Best Fire Detection: {best_fire_detection['experiment']} ({best_fire_detection['fire_f1_score']:.4f})")
        print(f"   ⚖️ Most Balanced: {best_balanced['experiment']} ({best_balanced['balanced_f1_score']:.4f})")
        print(f"\n📊 PERFORMANCE STATISTICS:")
        print(f"   📈 Accuracy Range: {performance_stats['accuracy_range'][0]:.4f} - {performance_stats['accuracy_range'][1]:.4f}")
        print(f"   📈 Average Accuracy: {performance_stats['accuracy_mean']:.4f} ± {performance_stats['accuracy_std']:.4f}")
        print(f"   🔥 Fire F1 Range: {performance_stats['fire_f1_range'][0]:.4f} - {performance_stats['fire_f1_range'][1]:.4f}")
        print(f"   ⏱️ Average Training Time: {performance_stats['avg_training_time_minutes']:.1f} minutes")
        print(f"\n💾 COMPREHENSIVE RESULTS SAVED:")
        print(f"   📄 Main Analysis: {final_summary_file}")
        print(f"   📊 CSV Reports: {csv_dir}/")
        print(f"   🗂️ Individual Results: {self.results_path}/experiment_details/")
        print(f"   🤖 Trained Models: {self.results_path}/models/")
        if self.bucket_name:
            print(f"\n📤 Uploading results to Google Cloud Storage...")
            upload_success = upload_results_to_gcs(self.bucket_name, str(self.results_path))
            if upload_success:
                print(f"✅ Results uploaded to gs://{self.bucket_name}/training_results/")
            else:
                print(f"❌ Failed to upload results to GCS")
        return final_summary
