# Synoptix: Synthetic Data for AI Fire Detection

This project investigates the use of synthetic data to train AI models for early fire detection in industrial environments. It uses Yolo on Google Cloud  for efficient training and evaluates optimal real/synthetic data ratios.

## Features
- Modular Python codebase for training, evaluation, and visualization.
- Supports GCP TPU/GPU/CPU fallback.
- Comprehensive results: Models, plots, JSON reports, CSV exports.
- Configurable via YAML.
- Docker support for reproducible runs.
- Basic tests and CI pipeline.

## Setup
1. Clone the repo: `git clone https://github.com/yourusername/synoptix-synthetic-fire-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
   - Or use Conda: `conda env create -f environment.yml && conda activate synoptix`
3. Prepare data: Place your datasets in `data/ratio_experiments/` (structure: e.g., `100R0S/train/fire/`, etc.). You manage this.
4. Configure: Edit `config/config.yaml` as needed.

## Usage
Run the training pipeline:
```bash
python src/main.py --bucket your-gcs-bucket --download-data --epochs 25 --fine-tune-epochs 10
```
- Args: See `python src/main.py --help`.
- Outputs saved to `results/`.

# Synoptix Synthetic Fire Detection

## Overview
This project provides a complete pipeline for training, evaluating, and visualizing fire detection models using synthetic and real datasets. It leverages the YOLO (You Only Look Once) object detection framework and includes tools for dataset management, model training, evaluation, and research-grade visual documentation.

## Features
- End-to-end training pipeline for fire detection using YOLOv8
- Support for mixed synthetic and real datasets with configurable ratios
- Automated visual documentation: sample image grids, dataset statistics, and composition plots
- Research evaluation scripts for statistical analysis and publication-ready plots
- Modular codebase for easy extension and experimentation

## Project Structure
```
├── Dockerfile
├── environment.yml
├── requirements.txt
├── README.md
├── config/
│   └── config.yaml
├── data/
│   ├── annot2.py
│   ├── annotation.py
│   ├── data_ratio_improved.py
│   ├── data_ratio.py
│   └── ...
├── docs/
│   └── project_brief.md
├── results/
├── src/
│   ├── __init__.py
│   ├── image_inference.py
│   ├── inference.py
│   ├── main.py
│   ├── metrics.py
│   ├── research_evalution.py
│   ├── trainer.py
│   ├── utils.py
│   └── ...
├── tests/
│   ├── __init__.py
│   └── test_trainer.py
```

## Setup
1. **Clone the repository:**
    ```bash
    git clone https://github.com/millind-yadav/synoptix-synthetic-fire-detection.git
    cd synoptix-synthetic-fire-detection
    ```
2. **Install dependencies:**
    - Using Conda (recommended):
       ```bash
       conda env create -f environment.yml
       conda activate synoptix-fire
       ```
    - Or using pip:
       ```bash
       pip install -r requirements.txt
       ```
3. **Prepare datasets:**
    - Place your datasets in the appropriate directory structure (see below).
    - Each dataset split (e.g., `synthetic_00pct`, `synthetic_10pct`, etc.) should contain a `data.yaml` file.

## Usage
- **Run the complete training pipeline:**
   ```bash
   python src/final_eval1.py
   ```
- **Train and evaluate models:**
   - Edit configuration in `config/config.yaml` as needed.
   - Run training and evaluation scripts in `src/`.
- **Generate research plots and analysis:**
   ```bash
   python src/research_evalution.py
   ```

## Dataset Structure
```
/home/milind/dataset/Balanced_Ratio_Experiment/
      synthetic_00pct/
            data.yaml
            images/
            labels/
      synthetic_10pct/
            data.yaml
            images/
            labels/
      ...
```

## Key Scripts
- `src/final_eval1.py`: Main pipeline for training, validation, and visual documentation
- `src/research_evalution.py`: Statistical analysis and research plots
- `src/trainer.py`: Model training utilities
- `src/metrics.py`: Evaluation metrics
- `src/utils.py`: Helper functions

## Testing
Run unit tests with:
```bash
pytest tests/
```

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, please open an issue or contact the repository owner.
