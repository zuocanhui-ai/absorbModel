# Toward Robust and Generalizable Frameworks: State-Space Deep Models for Predicting Uranium Adsorption on Biochar

Official implementation of the manuscript submitted to *Computational Materials Science*.

---

## Project Description

This repository provides the code and datasets for predicting uranium adsorption capacity (Qe) on biochar using machine learning and deep learning models. The primary model is a **Mamba-based state-space architecture with dropout regularization**, which is compared against eight baseline methods including tree-based ensembles, linear regression, feedforward neural networks, and attention-enhanced variants.

## Paper Information

- **Journal:** Computational Materials Science (Elsevier)
- **Status:** Submitted

## Repository Structure

```
.
├── data/                                        # Datasets
│   ├── REAL_biochar_adsorption_ECs_mapped.csv   # Full dataset with metadata columns
│   ├── Data_1267.csv                            # Archived dataset (not used)
│   └── Data.csv                                 # Curated dataset (1993 samples)
├── models/                                      # All model scripts
│   ├── Mamba_dropout_improved.py                # ★ Primary model (Mamba + Dropout)
│   ├── Mamba_improved.py                        # Mamba baseline
│   ├── Mamba_attention.py                       # Mamba + Attention
│   ├── FNN_improved.py                          # Feedforward Neural Network
│   ├── FNN_attention_improved.py                # FNN + Attention
│   ├── Random_forest_improved.py                # Random Forest
│   ├── GBDT_improved.py                         # Gradient Boosting
│   ├── Decsion_tree_improved.py                 # Decision Tree
│   ├── RegressionModel.py                       # Linear Regression
│   ├── Decsion_prediction.py                    # DT prediction export
│   ├── Model_comparsion.py                      # 9-model comparison (Data.csv)
│   └── Model_comparsion_2data.py                # 9-model comparison (Data.csv)
│   └── DataDistribution.py                      # Feature distribution visualization
├── figures/                                     # Generated figures
├── prediction/                                  # Prediction result files
├── requirements.txt                             # Python dependencies
├── LICENSE                                      # MIT License
└── README.md
```

## Dataset Description

### Input Features (10 dimensions)

| Feature | Unit | Description |
|---------|------|-------------|
| SA | m²/g | Specific surface area |
| Dav | nm | Average pore diameter |
| VTot | cm³/g | Total pore volume |
| C | wt% | Carbon content |
| O/C | — | Oxygen-to-carbon atomic ratio |
| (O+N)/C | — | Oxygen-plus-nitrogen-to-carbon atomic ratio |
| pH | — | Solution pH |
| T | K | Temperature |
| C0 | mg/L | Initial concentration |
| SLR | g/L | Solid-to-liquid ratio |

### Prediction Target

- **Qe (mg/g):** Equilibrium adsorption capacity

### Available Datasets

| File | Samples | Extra Columns | Used In |
|------|---------|---------------|---------|
| `Data.csv` | 1,993 | — | Model comparison (Fig. in manuscript) |
| `REAL_biochar_adsorption_ECs_mapped.csv` | 2,021 | Adsorbent, Pollutant | Individual model training |

The dataset used in the manuscript is **`Data.csv`**.

## Model Description

### Primary Model

**`models/Mamba_dropout_improved.py`** — Mamba with Dropout

This is the main model proposed in the manuscript. It uses a state-space block architecture (MambaBlock) with dropout regularization for robust uranium adsorption prediction. The model is trained with AdamW optimizer and uses GELU activations with residual connections.

### Baseline Models

| Model | Script | Category |
|-------|--------|----------|
| Linear Regression | `RegressionModel.py` | Linear |
| Decision Tree | `Decsion_tree_improved.py` | Tree-based |
| Random Forest | `Random_forest_improved.py` | Ensemble |
| GBDT | `GBDT_improved.py` | Ensemble |
| FNN | `FNN_improved.py` | Neural Network |
| FNN + Attention | `FNN_attention_improved.py` | Neural Network |
| Mamba | `Mamba_improved.py` | State-Space |
| Mamba + Attention | `Mamba_attention.py` | State-Space |

All models share the same data preprocessing pipeline (80/20 train-test split, StandardScaler normalization, random_state=42).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

See [requirements.txt](requirements.txt) for the full list. Key dependencies:

- Python >= 3.11
- PyTorch >= 2.0
- scikit-learn >= 1.3
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.8
- scipy >= 1.11

## Training Instructions

Run scripts from the **repository root directory**:

```bash
# Primary model (Mamba + Dropout)
python models/Mamba_dropout_improved.py

# Baseline models
python models/Mamba_improved.py
python models/FNN_improved.py
python models/Random_forest_improved.py
python models/GBDT_improved.py
python models/Decsion_tree_improved.py
python models/RegressionModel.py
python models/Mamba_attention.py
python models/FNN_attention_improved.py


Output: `figures/model_comparison_r2_scatter_3x3.png`

## Data Distribution

Generate feature distribution comparison (standardized vs. non-standardized):

```bash
python models/DataDistribution.py
```

Output: `figures/data_distribution_comparison_5x2.png`

## Output Files

| File | Generated By | Description |
|------|-------------|-------------|
| `figures/model_comparison_r2_scatter_3x3.png` | Model_comparsion*.py | 9-model R² scatter plot |
| `figures/FNN_convergence_curve.png` | FNN_improved.py | FNN training convergence |
| `figures/FNN_Attention_analysis.png` | FNN_attention_improved.py | FNN attention weights |
| `figures/Mamba_convergence_curve.png` | Mamba_improved.py | Mamba training convergence |
| `figures/Mamba_Dropout_convergence_curve.png` | Mamba_dropout_improved.py | Mamba+Dropout convergence |
| `figures/Mamba_Attention_analysis.png` | Mamba_attention.py | Mamba attention weights |
| `figures/data_distribution_comparison_5x2.png` | DataDistribution.py | Feature distributions |
| `prediction/DT_prediction_results.xlsx` | Decsion_prediction.py | DT prediction export |

## Reproducibility

All experiments can be reproduced using the provided datasets and scripts with the following guarantees:

- **Fixed random seeds:** `random_state=42` is used in all train-test splits and model initializations.
- **Default hyperparameters:** All scripts use the hyperparameters reported in the manuscript by default. No additional configuration is required.
- **Deterministic preprocessing:** StandardScaler is fit on the training set only and applied to the test set consistently across all models.
- **GPU/CPU compatibility:** PyTorch models automatically detect and use CUDA if available, falling back to CPU otherwise.

## Citation

Citation information will be updated after publication.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or issues, please open an issue on the GitHub repository.
