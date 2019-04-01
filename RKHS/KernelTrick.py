import numpy as np
import functions as fn
from sklearn.kernel_ridge import KernelRidge

class RKHS:
    def __init__(self, krn, l):
        self.krn = krn
        self.l = l
    def fit(self, dat):
        X,Y = fn.time_lagged_ts(dat)
        kr = KernelRidge(alpha = self.l, kernel=self.krn)
        krf = kr.fit(X, Y)
        return(krf)
    def predict(self, krf, Xx, horizon):
        out = []
        prd = Xx[np.shape(Xx)[0]-1,:]
        for n in range(horizon):
            prd = np.matrix(prd)
            prd = krf.predict(prd)
            out.append(prd)
        return(out)

