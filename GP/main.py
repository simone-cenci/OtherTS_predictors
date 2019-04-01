#%%
import importlib
import functions as fn
import numpy as np
import cv as cv
import GP as GSP
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
import matplotlib.pylab as plt
importlib.reload(fn)
importlib.reload(cv)
importlib.reload(GSP)
import scipy.stats as stat


#%%
ts = np.loadtxt('input/deterministic_chaos_k.txt')
#ts = np.loadtxt('input/empirical_time_series.txt')[:,1:5]
length_training = 180
training_set = ts[0:length_training,:]
### If you are running rolling window cross validation you need the unscale data
unscaled_training_set = training_set
training_set, scaler = fn.scale_training_data(training_set)
parameters = ParameterGrid({'l': np.logspace(0,1,15),
                            'kernel': [Matern]})



#%%
### Or Rolling window cross validation:
e, l, krn = cv.rollingcv(parameters, unscaled_training_set, 20, par = False)


#%%
#### Out of sample statistics
orizzonte = 20
gp_object = GSP.GP(krn(nu=l))
gpr = gp_object.fit(training_set)
pred, sigma = gp_object.predict(gpr, training_set, orizzonte)
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

#plt.fill_between(np.linspace(0, orizzonte, orizzonte-1), unscaled_pred - 1.96*np.squeeze(sigma), unscaled_pred+1.96*np.squeeze(sigma), alpha = 0.5)
print('Out of sample correlation:', corr)
print('Out of sample rmse:', rmse_test )
plt.show()
