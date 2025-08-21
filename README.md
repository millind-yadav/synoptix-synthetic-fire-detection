# Synoptix: Synthetic Data for AI Fire Detection

This project investigates the use of synthetic data to train AI models for early fire detection in industrial environments. It uses ResNet50 on Google Cloud TPU for efficient training and evaluates optimal real/synthetic data ratios.

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

For Docker:
```bash
docker build -t synoptix .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results synoptix --bucket your-bucket
```

## Project Brief
See [docs/project_brief.md](docs/project_brief.md) for details.

## Contributing
- Fork and PR.
- Run tests: `pytest tests/`
- Lint: `black src/ tests/`

## License
MIT
