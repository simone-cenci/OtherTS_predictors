#%%
import importlib
import functions as fn
import numpy as np
import cv as cv
import KernelTrick as RK
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
import matplotlib.pylab as plt
importlib.reload(fn)
importlib.reload(cv)
importlib.reload(RK)
import scipy.stats as stat


#%%
ts = np.loadtxt('input/deterministic_chaos_lv.txt')
#ts = ts + ts*np.random.normal(0,0.1*np.std(ts), size = (np.shape(ts)))
#ts = np.loadtxt('input/empirical_time_series.txt')[:,1:5]
length_training = 400
training_set = ts[0:length_training,:]
### If you are running rolling window cross validation you need the unscale data
unscaled_training_set = training_set
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'b': np.logspace(0,1,15),
                            'kernel': [Matern], ### Keep this kernel
                            'lambda': np.logspace(-3,-2,15)})



#%%
### Or Rolling window cross validation:
e, lmb, b, krn = cv.rollingcv(parameters, unscaled_training_set, 20, par = False)


#%%
#### Out of sample statistics
orizzonte = 150
rk_object = RK.RKHS(krn(b), lmb)
rkf = rk_object.fit(training_set)
pred = rk_object.predict(rkf, training_set, orizzonte)
pred = np.array(pred).reshape(orizzonte, np.shape(ts)[1])


#%%

### Scale back the prediction using the mean and standard deviation of the training set
unscaled_pred = fn.unscale_test_data(pred, scaler)
test_data = ts[length_training:(length_training+orizzonte),:]

corr = np.mean([stat.pearsonr(unscaled_pred[:,n],test_data[:,n])[0] 
                for n in range(np.shape(pred)[1])])
rmse_test = fn.rmse(test_data,unscaled_pred)
fig = plt.figure(figsize = (10,6))
for sp in range(np.shape(test_data)[1]):
    number='23'+str(sp+1)
    plt.subplot(number)
    titolo='Species '+str(sp+1)
    plt.title(titolo)
    plt.plot(test_data[:,sp], color = 'b', label = 'Data')
    plt.plot(unscaled_pred[:,sp], color = 'r', label = 'Prediction')
    if sp ==0:
        plt.legend()

print('Out of sample correlation:', corr)
print('Out of sample rmse:', rmse_test )
plt.show()
