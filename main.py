import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from dpmeans import *

def main():
    X = np.random.randn(200, 2)
    lam = 1
    epochs = 10

    model = dpmeans(X, lam)
    S, k = model.train(epochs)
    model.plot_2d(S)

if __name__ == '__main__':
    main()