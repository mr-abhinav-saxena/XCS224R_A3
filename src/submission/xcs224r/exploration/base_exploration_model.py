import numpy as np
from typing import Any, Dict

class BaseExplorationModel(object):
    def update(self, ob_no: np.ndarray, ac_na: np.ndarray, next_ob_no: np.ndarray, re_n: np.ndarray, terminal_n: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError