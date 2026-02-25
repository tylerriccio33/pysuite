# pysuite

A minimalist model evaluation tool that automatically generates performance reports for regression and classification tasks.

## Quick Start

### Install from GitHub

```bash
pip install git+https://github.com/yourusername/pysuite.git
```

### Example

```python
import polars as pl
from pysuite import run, show

# Create sample data
xeval = pl.DataFrame({
    "feature_1": [1.0, 2.0, 3.0],
    "feature_2": [4.0, 5.0, 6.0]
})
yeval = pl.Series([10.0, 20.0, 30.0])
ypred = pl.Series([10.5, 19.8, 30.2])

# Get report as dict
report = run(xeval, yeval, ypred)

# Launch web interface — pick your style:
run(xeval, yeval, ypred, show=True)  # functional
report.show()                         # method on result
show(report)                          # standalone function
```

## What It Does

- **Auto-detects** regression vs classification
- **Calculates metrics** (MSE, MAE, R² for regression; Accuracy, Precision, Recall, F1 for classification)
- **Creates a results table** with predictions, actuals, residuals, and features
- **Optionally displays** a Flask web interface for viewing reports

Works with Polars, Pandas, and other dataframe libraries via [narwhals](https://github.com/narwhals-dev/narwhals).
