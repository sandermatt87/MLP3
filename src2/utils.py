import numpy as np

#utility functions
def to_single_class(labels):
	print( "transforming")
	print( labels)
	new_labels=np.zeros(labels.shape[0])
	for i in range(0,labels.shape[1]):
			new_labels=new_labels+labels[:,i]*2**i
	print( new_labels)
	return new_labels
	
def to_multiple_classes(labels,nclasses):
	print( "backtransforming")
	print( labels)
	new_labels=np.zeros((labels.shape[0],nclasses))
	tmp=np.copy(labels)
	for i in range(0,nclasses):
		new_labels[:,i]=tmp % 2
		tmp=tmp-new_labels[:,i]
		tmp/=2
	print( new_labels)
	return new_labels
