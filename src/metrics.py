from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr
