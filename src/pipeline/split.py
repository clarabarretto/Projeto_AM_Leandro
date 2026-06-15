"""Split estratificado 70/30 com seed fixa."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.30


def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split 70/30 estratificado por classe."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
