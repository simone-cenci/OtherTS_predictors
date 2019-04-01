import numpy as np
import functions as fn
from sklearn.gaussian_process import GaussianProcessRegressor

class GP:
    def __init__(self, krn):
        self.krn = krn
    def fit(self, dat):
        X,Y = fn.time_lagged_ts(dat)
        gp = GaussianProcessRegressor(kernel=self.krn, n_restarts_optimizer=0)
        gpr = gp.fit(X, Y)
        return(gpr)
    def predict(self, gpr, Xx, horizon):
        out = []
        std = []
        prd = Xx[np.shape(Xx)[0]-1,:]
        for n in range(horizon):
            prd = np.matrix(prd)
            prd, sigma = gpr.predict(prd, return_std=True)
            out.append(prd)
            std.append(sigma)
        return(out,std)