# Advanced ML-Enhanced Test Failure Prediction Framework

![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning-driven framework for predicting software test failures that combines automated data collection from open-source repositories, advanced ML models, and validation experiments on real-world datasets.

This framework was developed as part of research into software reliability and predictive maintenance, with results presented in the accompanying research paper.

---

## 📌 Features

* **Automated Data Collection** – Seamlessly gather data from GitHub repositories  
* **ML-Enhanced Prediction** – Advanced machine learning models for accurate test failure detection  
* **Real-World Validation** – Comprehensive validation pipeline tested on production projects  
* **Research-Backed** – Methodology supported by extensive analysis and visualizations  
* **Reproducibility** – Demo dataset, pinned dependencies, and reproducible experiment scripts

---

## 🚀 Quick Start (Demo)

Run the demo pipeline in under 2 minutes:

```bash
git clone https://github.com/zolydiac/Advanced-ML-Enhanced-Test-Failure-Prediction-Framework.git
cd Advanced-ML-Enhanced-Test-Failure-Prediction-Framework

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run demo with small synthetic dataset
python main.py --demo

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/zolydiac/Advanced-ML-Enhanced-Test-Failure-Prediction-Framework.git
cd Advanced-ML-Enhanced-Test-Failure-Prediction-Framework
pip install -r requirements.txt
```

## 📊 Two Validation Approaches

### 1. Reproducible Demo (`main.py --demo`)
- **Purpose**: Academic validation, research reproducibility
- **Data**: Synthetic dataset with controlled parameters
- **Reproducibility**: Always generates identical results (seeded random generation)
- **Use Case**: Proving methodology, academic papers, controlled experiments

### 2. Real-World Validator (`enhanced_real_world_validator.py`)
- **Purpose**: Practical validation, deployment readiness
- **Data**: Live GitHub repository data via API
- **Reproducibility**: Results vary as repositories evolve
- **Use Case**: Demonstrating real-world applicability, production validation

Both approaches complement each other - the synthetic data proves the methodology works under controlled conditions, while real-world data shows it handles messy, practical scenarios.

**(Optional: create a virtual environment before installing.)**

## 🖥️ Usage

Run validation on target repositories:

```bash
python src/validation/enhanced_real_world_validator.py --repos microsoft/playwright pytest-dev/pytest --days 60
```

General usage format:

```bash
python src/validation/enhanced_real_world_validator.py --repos <org/repo1> <org/repo2> --days <N>
```

## 📂 Project Structure

```
.
├── src/                    # Core framework code
│   ├── validation/         # Validation pipelines  
│   ├── models/             # ML model implementations
│   └── utils/              # Helper functions
├── paper/                  # Research paper & figures
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## 🧪 Running Tests

Run unit tests with pytest:

```bash
pytest -q
```

## 📊 Research Paper

The detailed research, including methodology, datasets, results, and visualizations, is available in the `paper/` directory.

## 🛠️ Contributing

Contributions are welcome!

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/new-idea`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/new-idea`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

## ⭐ Acknowledgments

* Python, scikit-learn, and PyTorch ecosystems
* Open-source projects providing datasets
* Guidance from academic research on software reliability
jjjjjjjjjjjj