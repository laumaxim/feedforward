import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as pyplot

X,y = make_moons(n_samples=50, noise=0.2, random_state=43)
X_bias = np.hstack((X,np.ones((50,1))))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unit_test_sigmoid():
    '''unit test of function sigmoid()'''
    a = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    expected = np.array([0.0, 0.27, 0.5, 0.73, 1.0])
    assert np.all(sigmoid(a).round(2) == expected)


def log_loss(ytrue, ypred):
    loss = - (ytrue*np.log(ypred) + (1-ytrue)*np.log(1-ypred))
    return loss



def feed_forward(X, weights:list, activation=sigmoid):
    """
    1. Calculate the dot product of X
       and the weights of the first layer.

    2. Apply the sigmoid function on the result.

    3. Append an extra column of ones to the result (i.e. the bias).

    4. Calculate the dot product of the previous step
       with the weights of the second (i.e. outer) layer.

    5. Apply the sigmoid function on the result.

    6. Return all intermediate results (i.e. anything that is outputted
       by an activation function).
    """
    
    output = []
    
    for w in weights[:-1]:
        step1 = np.dot(X,w)
        step2 = activation(step1)
        output.append(step2)
        step3 = np.hstack((step2,np.ones(shape=(step2.shape[0],1))))
        X = step3
        
    outer_weights = weights[-1]
    step4 = np.dot(X,outer_weights)
    step5 = activation(step4)
    output.append(step5)

    return output


def unit_test_ff():
    X_bias = np.hstack((X,np.ones((50,1))))
    hidden_weights = np.random.uniform(size=(3,2))
    outer_weights = np.random.uniform(size=(3,1))
    out = feed_forward(X_bias, [hidden_weights, outer_weights])
    assert out[0].shape == (50, 2)
    assert out[0].shape == (50, 1)

    Xref = np.array([[1.0, 2.0, 1.0]])
    whidden = np.array([[1.0, 2.0, 0.0],
                    [-1.0, -2.0, 0.0]
                        ]).T
    wout = np.array([1.0, -1.0, 0.5]).T

    out = feed_forward(Xref, [whidden, wout])
    assert np.all(out[0].round(2) == np.array([[0.99, 0.01]]))
    assert np.all(out[1].round(2) == np.array([[0.82]]))


def unit_test_logloss():
    ytrue = np.array([0.0, 0.0, 1.0, 1.0])
    ypred = np.array([0.01, 0.99, 0.01, 0.99])
    expected = np.array([0.01, 4.61, 4.61, 0.01])
    assert np.all(log_loss(ytrue, ypred).round(2) == expected)

def backprop(weights,
             output,
             ytrue,
             X_input,
             LR_O,
             LR_H):
    #separate learning rates for outer and inner weights.

    output.insert(0, X_input)
    
    w_new = []
    
    #Start with the last layer, the output layer
    '''EQUATION A:'''
    ytrue = ytrue.reshape(-1, 1)
    error = (output[-1] - ytrue) * log_loss(ytrue, output[-1])
    '''EQUATION B:'''
    sig_deriv = output[-1] * ( 1 - output[-1])
    #derivative of the sigmoid function with respect to the
    #hidden output * weights 
    grad = sig_deriv * error
    
    '''EQUATION C:'''
    bias = np.ones((output[-2].shape[0] ,1))
    hidden_out_bias = np.hstack([output[-2],bias])
    #don't forget the bias!
    delta_wo = np.dot(- grad.transpose(), hidden_out_bias) * LR_O
    
    #and finally, old weights + delta weights -> new weights!
    w_new.append(weights[-1] + delta_wo.transpose())
    
    # l=2 is the second-last layer
    for l in range(2,len(output)):
        #print(f'l: {l}')
        '''EQUATION D:'''
        sig_deriv = output[-l] * ( 1 - output[-l])
        w = weights[-l+1]
        #exclude the bias (3rd column) of the outer weights,
        #since it is not backpropagated!
        ## ??
        #grad = np.dot(grad,w[:-1].transpose())
        grad = sig_deriv  * np.dot(grad,w[:-1].transpose())
        '''EQUATION E:'''
        bias =  np.ones((output[-l-1].shape[0] ,1))
        z = np.hstack([output[-l-1],bias])
        delta_wH = np.dot(-grad.transpose(), z)*LR_H
        w_new.append(weights[-l] + delta_wH.transpose())
    return list(reversed(w_new))

def two_layers():
    LOSS_VEC = []
    ACC_VEC = []
    hidden_weights = np.random.uniform(size=(3,2)) # 3 inputs get passed to 2 neurons
    outer_weights = np.random.uniform(size=(3,1))
    weights = [hidden_weights,outer_weights]
    for _ in range(1000):
        out = feed_forward(X_bias, weights)
        LOSS_VEC.append(sum(log_loss(y.reshape(-1,1),out[-1]))[0])
        ypred=out[-1].round()
        ACC_VEC.append(sum((ypred) == y.reshape(-1,1)) / len(y))
        new_weights = backprop(weights, out, y, X,0.01,0.01)
        weights = new_weights



def four_layers():
    weights1 = np.random.uniform(size=(3,5))
    weights2 = np.random.uniform(size=(6,4))
    weights3 = np.random.uniform(size=(5,5))
    weights4 = np.random.uniform(size=(6,2))
    outer_weights = np.random.uniform(size=(3,1))
    #weights = [weights1,outer_weights]
    weights = [weights1,weights2,weights3,weights4,outer_weights]
    LOSS_VEC = []
    ACC_VEC = []
    for _ in range(400):
        #print(f'iteration {i}')
        out = feed_forward(X_bias, weights)
        LOSS_VEC.append(sum(log_loss(y.reshape(-1,1),out[-1]))[0])
        ypred=out[-1].round()
        ACC_VEC.append(sum((ypred) == y.reshape(-1,1)) / len(y))
        new_weights = backprop(weights, out, y, X,0.01,0.01)
        weights = new_weights