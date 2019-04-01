import numpy as np

import functions as fn
from operator import itemgetter
from joblib import Parallel, delayed
import multiprocessing
import GP as GSP
from sklearn.gaussian_process.kernels import RBF, Matern
import importlib
importlib.reload(fn)

def fun_rolling(i, grid, dat, orizzonte):
	'''
	This is for convenience so that if you want to run 
	any rolling cross validation in parallel you can
	do it easily
	'''
	iterations = 10
	#r = GSP.GP(Matern(grid[i]['l']))
	r = GSP.GP(grid[i]['kernel'](grid[i]['l']))
	val_err = []
	for n in range(iterations):
		tr_dat = dat[0:(np.shape(dat)[0]-iterations+n-orizzonte),:]
		tr_dat, scaler_cv = fn.scale_training_data(tr_dat)
		gpr = r.fit(tr_dat)
		prd, _ = r.predict(gpr,tr_dat, orizzonte)
		prd = fn.unscale_test_data(prd, scaler_cv)
		val_dat = dat[(np.shape(dat)[0]-iterations+n-orizzonte):(np.shape(dat)[0]-iterations+n),:]
		val_err.append(fn.rmse(val_dat,prd))
	return(np.mean(val_err))

def rollingcv(grid,dat, orizzonte, par = False):
	'''
	Here I run rolling window cross validation. That is:
	calling ---- the training data and **** the validation data
	we iteratively run:
	iter 1.) -------****
	iter 2.) --------****
	iter 3.) ---------****
	iter 4.) ----------****
	Then we take the validation error over the iterations
	'''
	if par:
		num_cores = multiprocessing.cpu_count() - 3
		print('Running in parallel')
		error = Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(fun_rolling)(i, grid,dat, orizzonte) for i in range(len(grid)))
	else:
		print('Running not in parallel: safest choice at the moment')
		error = [fun_rolling(i, grid,dat, orizzonte) for i in range(len(grid))]
	idx = min(enumerate(error), key=itemgetter(1))[0]
	return(error, grid[idx]['l'], grid[idx]['kernel'])

