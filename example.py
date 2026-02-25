"""Example: Run the Pysuite reporting server with sample data."""

import polars as pl

from pysuite import run


def main() -> None:
    """Run the Pysuite reporting server with sample data."""
    # Generate sample regression data
    n_samples = 100
    n_features = 5

    feature_data = {f"feature_{i}": [float(i) for i in range(n_samples)] for i in range(n_features)}
    xeval = pl.DataFrame(feature_data)

    y_true = [float(i) + 50 for i in range(n_samples)]
    y_pred = [yt + (i * 2) for i, yt in enumerate(y_true)]

    yeval = pl.Series(y_true)
    ypred = pl.Series(y_pred)

    # Run and show the web interface
    run(xeval, yeval, ypred, show=True)


if __name__ == "__main__":
    main()
