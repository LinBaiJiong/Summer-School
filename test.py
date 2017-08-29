import scipy.io as sio
import matplotlib.pyplot as plt  
import numpy as np 
from numpy import linalg
import pandas as pd


def LR_train( x , y , lamb=0.001 ):
    [m, d] = x.shape
    
    x = np.column_stack( ((np.ones(m)).T , x) )
    # add a*I(I: identity matrix) into x.T.dot(x) to make sure that x.T.dot(x) is a invertible matrix
    w = linalg.pinv(x.T.dot(x) + np.eye((d + 1)) * lamb).dot(x.T).dot(y)
    
    return w


def LR_test( x , y, w, problem_type='regression' ):
    m, d = x.shape
    
    x = np.column_stack( ((np.ones(m)).T , x) )
    t = np.dot(x, w)
    
    er = t - y
    MSE = (er**2).mean()                      
    MAE = abs(er).mean()                      
    SD = np.sqrt((( er-er.mean())**2).mean()) 
    
    if problem_type=='classification':
        acc = np.equal(np.rint(t), y).mean() * 100
        text = "The Linear Regression's accuracy is %.2f%%." %(acc)
        print(text)
    
    obj = pd.Series([MSE,MAE,SD], index=['MSE','MAE','SD'])
    print(obj)
    
    return t

# change path 
data_path = "data/SpectralClassification/No_removal_of_continuous_spectrum/"

data=sio.loadmat(data_path + "SpectralClassificationTrain.mat") 
train_x = data['train_x']
train_y = np.reshape(data['train_y'][:,0], [-1,1])

data=sio.loadmat(data_path + "SpectralClassificationTest.mat") 
test_x = data['test_x']
test_y = np.reshape(data['test_y'][:,0], [-1,1])
# add noise into data
noise_y = test_y + np.random.randn(200, 1)
del data


w = LR_train(train_x, train_y)
t = LR_test(test_x , test_y , w, problem_type='classification')

# show
m, d = test_x.shape
test_x = np.column_stack( ((np.ones(m)).T , test_x) )

plt.plot(test_x[:,1], noise_y, 'o')
plt.plot(test_x[:,1], np.dot(test_x, w))
plt.show()
