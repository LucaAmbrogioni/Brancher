
import numpy as np

from brancher.utilities import coerce_to_dtype

x = np.random.normal(size=(10, 3))
T = coerce_to_dtype(x)