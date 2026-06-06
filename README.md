<div align="center">

# Synthetic Data Artist
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-VAE-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-green) ![Status](https://img.shields.io/badge/Status-Research%20Demo-purple) [![CI](https://github.com/AmirhosseinHonardoust/Synthetic-Data-Artist/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Synthetic-Data-Artist/actions/workflows/ci.yml)
</div>

A professional research-style Python project for generating and evaluating **synthetic tabular data**. The project compares a **Gaussian Copula** generator with a lightweight **Variational Autoencoder (VAE)** and evaluates the generated data using distribution, correlation, categorical similarity, boundary validity, privacy-proxy, and optional downstream machine-learning utility checks.

> **Important:** This project is a **research and portfolio demo**, not a certified privacy-preserving synthetic data product.
>
> It can help analyze synthetic data quality, but it does not provide formal differential privacy or guarantee that generated records are safe to release.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Features](#features)
- [Methods](#methods)
- [Charts and Visual Analysis](#charts-and-visual-analysis)
- [How the Evaluation Works](#how-the-evaluation-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Generator](#running-the-generator)
- [Command-Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Generated Outputs](#generated-outputs)
- [Evaluation](#evaluation)
- [Privacy Proxy Analysis](#privacy-proxy-analysis)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Synthetic data generation is useful when teams want to experiment, prototype, share examples, or test workflows without exposing raw sensitive datasets. However, synthetic data is often misunderstood. Generating fake-looking rows does not automatically make a dataset private, useful, or statistically realistic.

This project takes a more careful approach. It does not only generate synthetic rows; it also evaluates how closely the synthetic data preserves important properties of the original table.

The goal of this project is to demonstrate:

- A clean synthetic-data generation workflow
- Statistical and neural synthetic-data methods
- Honest quality evaluation
- Privacy-risk proxy diagnostics
- Optional train-on-synthetic, test-on-real utility evaluation
- Visual reporting
- Configurable CLI execution
- Tests and CI for reproducibility
- Clear limitations and responsible-use documentation

---

## What This Project Does

This project can:

- Load a real tabular CSV dataset
- Generate synthetic rows using a Gaussian Copula method
- Generate synthetic rows using a lightweight VAE method
- Detect numeric and categorical columns automatically
- Preserve the original column structure in synthetic outputs
- Validate input data and configuration before generation
- Generate quality metrics in JSON format
- Generate a one-row `quality_summary.csv` for quick comparison
- Create visual diagnostics for distributions, PCA, correlations, and pairplots
- Produce lightweight HTML reports
- Compute privacy proxy diagnostics such as exact duplicate rate and nearest-neighbor distances
- Optionally evaluate downstream ML utility when a target column is configured
- Run automated tests and CI smoke workflows

---

## What This Project Does Not Do

This project does **not**:

- Provide formal differential privacy
- Certify that synthetic data is safe to publish
- Guarantee that generated records cannot leak information
- Replace domain-specific privacy review
- Replace mature libraries such as SDV, CTGAN, or commercial privacy platforms
- Guarantee strong performance on every dataset
- Prove that a synthetic dataset is suitable for high-stakes use

A production-grade synthetic-data system would require stronger schema constraints, formal privacy evaluation, domain validation, monitoring, and security review.

---

## Features

- **Gaussian Copula generator** for statistical synthetic tabular data
- **Variational Autoencoder generator** for neural synthetic tabular data
- **Automatic schema detection** for numeric and categorical columns
- **Configurable VAE hyperparameters**
- **Configurable output directories**
- **Input dataframe validation**
- **Config validation** with clear error messages
- **Distribution overlap metrics**
- **Correlation difference metrics**
- **Categorical distribution similarity**
- **Numeric summary-statistic differences**
- **Boundary violation checks**
- **Exact duplicate-rate check**
- **Nearest-neighbor privacy proxy metrics**
- **Optional ML utility evaluation**
- **PCA projection chart**
- **Correlation heatmap**
- **Distribution comparison chart**
- **Pairplot comparison chart**
- **HTML report generation**
- **Unit test suite**
- **GitHub Actions CI support**

---

## Methods

The project currently supports two synthetic-data generation methods.

### Gaussian Copula

The Gaussian Copula method models feature distributions and dependency patterns, then samples new rows from the fitted statistical structure.

It is often useful for small or medium-sized tabular datasets where statistical relationships are relatively stable.

### Variational Autoencoder

The VAE method learns a compressed latent representation of the dataset and decodes synthetic rows from that latent space.

In this repository, the VAE is intentionally lightweight and configurable. It should be treated as a baseline neural generator, not a fully tuned production VAE.

---

## Charts and Visual Analysis

The project automatically generates charts to make synthetic-data quality easier to inspect.

Generated charts are saved in:

```text
outputs/<run_name>/plots/
```

Main charts include:

<div align="center">

| Chart | Purpose |
|---|---|
| Distribution overlap | Compares numeric feature distributions between real and synthetic data |
| PCA projection | Shows whether real and synthetic rows occupy similar low-dimensional space |
| Correlation heatmap | Compares correlation structure between real and synthetic data |
| Pairplot comparison | Provides visual pairwise comparisons for sampled rows |
</div>

These charts are diagnostic tools. They help identify obvious quality problems, but they should not be treated as proof that synthetic data is private or production-ready.

---

## How the Evaluation Works

The evaluation workflow compares real and synthetic datasets across multiple dimensions:

```text
Real dataset
     ↓
Synthetic generator
     ↓
Synthetic dataset
     ↓
Quality metrics + visual diagnostics + optional ML utility evaluation
```

The evaluation includes:

- Numeric distribution similarity
- Numeric correlation preservation
- Categorical distribution similarity
- Summary-statistic differences
- Boundary validity checks
- Privacy proxy diagnostics
- Optional train-synthetic-test-real utility checks

This makes the project more useful than a generator-only demo, because it asks whether the generated data actually behaves like the original data.

---

## Project Structure

```text
Synthetic-Data-Artist/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── real_data.csv
│   └── synthetic_data_*.csv
│
├── outputs/
│   └── <run_name>/
│       ├── metrics.json
│       ├── quality_summary.csv
│       └── plots/
│           ├── distribution_overlap.png
│           ├── pca_projection.png
│           ├── correlation_heatmap.png
│           └── pairplot_comparison.png
│
├── reports/
│   └── <run_name>_report.html
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data.py
│   ├── schema.py
│   ├── models/
│   │   ├── copula.py
│   │   └── vae.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── plots.py
│   └── reporting/
│       └── html_report.py
│
├── tests/
│   ├── test_core_contracts.py
│   ├── test_project_integrity.py
│   ├── test_enhanced_evaluation_metrics.py
│   └── test_cli_and_validation.py
│
├── config.yaml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Synthetic-Data-Artist.git
cd Synthetic-Data-Artist
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Running the Generator

Run the Copula workflow:

```bash
python src/main.py --method copula --run_name copula_run
```

Run the VAE workflow:

```bash
python src/main.py --method vae --run_name vae_run
```

Validate configuration and input data without generating synthetic data:

```bash
python src/main.py --validate-only
```

Run a faster workflow without the pairplot:

```bash
python src/main.py --method copula --run_name fast_copula --skip-pairplot
```

---

## Command-Line Usage

Basic CLI example:

```bash
python src/main.py \
  --config config.yaml \
  --data data/real_data.csv \
  --method copula \
  --run_name copula_experiment
```

Available options:

<div align="center">

| Option | Description |
|---|---|
| `--config` | Path to YAML configuration file |
| `--data` | Path to real input CSV file |
| `--method` | Generation method: `copula` or `vae` |
| `--run_name` | Name used for output folders and files |
| `--rows` | Override the number of synthetic rows |
| `--outdir` | Override the root output directory |
| `--data-outdir` | Override the synthetic CSV output directory |
| `--report-dir` | Override the HTML report directory |
| `--skip-pairplot` | Skip pairplot generation for faster runs |
| `--validate-only` | Validate config/data/schema and exit |
</div>

Example with custom directories:

```bash
python src/main.py \
  --method vae \
  --run_name experiment_vae \
  --rows 500 \
  --outdir experiment_outputs \
  --data-outdir experiment_data \
  --report-dir experiment_reports \
  --skip-pairplot
```

---

## Configuration

Main configuration is stored in:

```text
config.yaml
```

Example configuration:

```yaml
rows: 1000
categorical_threshold: 20
seed: 42
pca_components: 2
hist_bins: 30
pairplot_sample: 500

paths:
  data_dir: data
  output_dir: outputs
  report_dir: reports

plots:
  pairplot: true

vae:
  epochs: 30
  batch_size: 128
  latent_dim: 8
  hidden_dim: 64
  learning_rate: 0.001
  kl_weight: 0.001

evaluation:
  privacy_max_rows: 500
  ml_utility:
    target: null
    test_size: 0.25
```

To enable ML utility evaluation, set a target column:

```yaml
evaluation:
  ml_utility:
    target: target
    test_size: 0.25
```

---

## Generated Outputs

Each run creates a synthetic CSV, metrics, plots, and an HTML report.

```text
data/synthetic_data_<run_name>.csv
outputs/<run_name>/metrics.json
outputs/<run_name>/quality_summary.csv
outputs/<run_name>/plots/distribution_overlap.png
outputs/<run_name>/plots/pca_projection.png
outputs/<run_name>/plots/correlation_heatmap.png
outputs/<run_name>/plots/pairplot_comparison.png
reports/<run_name>_report.html
```

### Output Files

<div align="center">

| File | Purpose |
|---|---|
| `synthetic_data_<run_name>.csv` | Generated synthetic dataset |
| `metrics.json` | Full structured evaluation metrics |
| `quality_summary.csv` | Compact one-row summary for comparing runs |
| `plots/` | Visual diagnostics |
| `<run_name>_report.html` | Lightweight HTML report |
</div>

---

## Evaluation

The project uses a multi-part evaluation workflow.

Evaluation includes:

- Distribution overlap
- Correlation difference
- Categorical similarity
- Numeric summary-statistic differences
- Boundary violation rate
- Privacy proxy diagnostics
- Optional ML utility evaluation

### Distribution Overlap

Measures how close numeric feature distributions are between real and synthetic data using Jensen-Shannon distance transformed into an overlap-style score.

Higher is better.

### Correlation Difference

Compares numeric correlation matrices between real and synthetic data.

Lower is better.

### Categorical Similarity

Compares category proportions between real and synthetic data using total-variation similarity.

Higher is better.

### Numeric Summary Difference

Compares scaled differences in numeric summary statistics such as mean, standard deviation, minimum, and maximum.

Lower is better.

### Boundary Violation Rate

Checks whether synthetic values fall outside observed real-data numeric ranges or create invalid categorical values.

Lower is better.

### ML Utility Evaluation

When a target column is configured, the project compares:

```text
train on real data      → test on held-out real data
train on synthetic data → test on held-out real data
```

This helps estimate whether synthetic data preserves downstream predictive utility.

---

## Privacy Proxy Analysis

The project includes lightweight privacy-risk proxy diagnostics.

These include:

- Exact duplicate rate
- Mean nearest-neighbor distance
- 5th percentile nearest-neighbor distance
- Minimum nearest-neighbor distance

These metrics help identify potential memorization or overly close synthetic records.

> **Important:** These are proxy diagnostics only. They do not prove privacy and should not be treated as a formal privacy guarantee.

---

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

Compile source and test files:

```bash
python -m compileall src tests
```

The tests check important project behavior, including:

- Schema detection
- Copula output contracts
- VAE output contracts
- Enhanced evaluation metrics
- Config validation
- Input dataframe validation
- CLI argument parsing
- Existing metrics JSON validity
- Requirements formatting
- Source compilation

---

## Code Quality

The project includes automated workflow checks through:

```text
.github/workflows/ci.yml
```

The CI workflow checks:

- Dependency installation
- Source compilation
- Unit tests
- Config and input validation
- Copula smoke workflow
- VAE smoke workflow
- Expected output files
- Required metrics in generated JSON files

This provides a basic reproducibility and regression safety net for future changes.

---

## Limitations

This project has important limitations.

The project:

- Uses demo data by default
- Does not provide formal differential privacy
- Does not certify that synthetic data is safe to publish
- Uses a lightweight baseline VAE
- May not preserve complex real-world relationships
- Uses proxy privacy diagnostics, not formal privacy proofs
- Requires domain-specific validation for real datasets
- May generate poor synthetic data if the input data is small, noisy, or highly constrained
- May be slow on larger datasets when pairplot generation is enabled

High quality scores on one dataset do not guarantee that the method will work well on another dataset.

---

## Responsible Use

This project is intended for:

- Synthetic data education
- Research-style experimentation
- Portfolio demonstration
- Data quality diagnostics
- Comparing simple synthetic-data generation methods
- Learning about synthetic-data evaluation workflows

It should not be used as-is for:

- Publishing synthetic data derived from sensitive records
- Healthcare, financial, legal, or high-stakes data release
- Replacing formal privacy review
- Claiming differential privacy
- Production synthetic-data deployment without additional safeguards

Before using synthetic data in sensitive contexts, evaluate duplicate rates, nearest-neighbor distances, domain constraints, utility metrics, and privacy risks with expert review.

---

## Future Improvements

Possible future improvements include:

- Add CTGAN or TVAE-style generators
- Add formal privacy evaluation methods
- Add richer schema metadata and constraints
- Add per-column quality cards
- Add train-synthetic-test-real benchmark reports
- Add support for larger benchmark datasets
- Add a Streamlit dashboard for visual comparison
- Add experiment tracking across multiple runs
- Add Docker support
- Add model artifact saving and loading
- Add more advanced missing-data handling
- Add configurable plot generation levels

---

## Tech Stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- PyTorch
- matplotlib
- seaborn
- PyYAML
- HTML reports
- unittest
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational, research, and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
