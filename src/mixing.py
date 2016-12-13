import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

#optimices the weights of a list of moodels so that they perform optimally under cross validiation
def cv_optimization(cv_predictions,targets,nmodel):
	ntrain=cv_predictions[:,0].shape
	def loss(weights):
		#normalize thei weights to ensure ther sum is 1
		norm_weights=np.copy(weights)
		norm_weights=np.insert(norm_weights, 0, 0.0001) #fix the first entry (before normalisation) so that the solution is unique (and positive)
		norm_weights=norm_weights/np.sum(norm_weights)
		predictions=np.zeros(ntrain)
		for i in range(0, nmodel):
			predictions = np.c_[predictions, norm_weights[i]*(cv_predictions[:,i])]
		predictions = np.sum(predictions, axis=1)
		#print log_loss(targets, predictions)
		return log_loss(targets, predictions)
	initial_weights = np.random.rand(nmodel-1)
	print nmodel
	optimal_weights = minimize(loss, initial_weights, method='L-BFGS-B', bounds=[(0, None)]*(nmodel-1)).x
	optimal_weights = np.insert(optimal_weights, 0, 0.0001)
	optimal_weights = optimal_weights/np.sum(optimal_weights)
	print "Mixed models with cv_optimization, resulting loss:", loss(optimal_weights)
	return optimal_weights
