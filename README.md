# SMS Spam Detection: IEEE Research Paper and Reproducible Code

This repository contains the complete source code, figures, experiment outputs,
and IEEE LaTeX manuscript for the paper:

**An Error-Sensitive and Reproducible Baseline for SMS Spam Detection Using
Bag-of-Words and Multinomial Naive Bayes**

## Author

**Aritrik Ghosh**  
Department of Computer Science and Engineering  
Swami Vivekananda University, India

## Repository structure

```text
.
├── main.tex
├── main.pdf
├── requirements.txt
├── experiment.py
├── benchmark.py
├── confidence_intervals.py
├── plot_results.py
├── run_experiment.sh
├── compile_paper.sh
├── figures/
│   ├── architecture.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── error_profile.png
│   └── results_metrics.png
├── results/
│   ├── results.csv
│   ├── confusion_matrix.npy
│   └── classification_report.txt
└── data/
    └── README.md
```

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Add the dataset

Place `spamraw.csv` in the repository root. It must contain:

- `type`: `ham` or `spam`
- `text`: SMS message

The original reported experiment used a supplied local CSV with 5,559 usable rows.
The repository does not redistribute the dataset.

## 3. Run the reported baseline

```bash
python experiment.py
python plot_results.py
python confidence_intervals.py
```

Or on Linux/macOS:

```bash
./run_experiment.sh
```

The experiment uses:

- `test_size=0.20`
- `random_state=0`
- no stratification
- `CountVectorizer()` with default parameters
- `MultinomialNB()` with default parameters

## 4. Reproduce the paper

The current result files are already included under `results/`. To regenerate the
paper figures after rerunning the experiment:

```bash
python experiment.py
python plot_results.py
```

To compile the IEEE paper:

```bash
pdflatex main.tex
pdflatex main.tex
```

Or:

```bash
./compile_paper.sh
```

## 5. Optional comparative benchmark

The repository also includes `benchmark.py`, which is an execution-ready follow-up
benchmark. It does **not** claim comparative results in the current paper because the
exact original CSV is not included here.

Run it with:

```bash
python benchmark.py spamraw.csv --output benchmark_results.csv
```

It evaluates six lightweight pipelines with repeated stratified 5-fold cross-validation:

- CountVectorizer + Multinomial Naive Bayes
- TF-IDF + Multinomial Naive Bayes
- Word 1-2 grams + Multinomial Naive Bayes
- Character 3-5 grams + Multinomial Naive Bayes
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

## Reported baseline result

The current manuscript reports the actual fixed hold-out experiment supplied by the
author:

- 5,559 messages
- 4,812 ham
- 747 spam
- 4,447 training messages
- 1,112 test messages
- Accuracy: 99.01%
- Spam precision: 100.00%
- Spam recall: 91.73%
- Spam F1: 95.69%
- Confusion matrix: `[[979, 0], [11, 122]]`

The paper explicitly treats these as results from one fixed non-stratified hold-out
partition, not as universal performance estimates.

## Important reproducibility note

The original run did not record exact Python/package versions or a checksum of the
local CSV. The manuscript identifies these as limitations rather than inventing them.
