import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def ReLU(z):
    return np.maximum(0,z)
  

def tanh(z):
    return np.tanh(z)


def log_loss(ytrue, ypred):
    loss = - (ytrue*np.log(ypred) + (1-ytrue)*np.log(1-ypred))
    return loss


def logloss_deriv(activation, y):
    return -y/activation + (1-y)/(1-activation)


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
    error = logloss_deriv(output[-1],ytrue) 
    '''EQUATION B:'''
    sig_deriv = output[-1] * ( 1 - output[-1])
    #derivative of the sigmoid function with respect to the
    #hidden output * weights 
    grad = sig_deriv * error
    
    '''EQUATION C:'''
    bias = np.ones((output[-2].shape[0], 1))
    hidden_out_bias = np.hstack([output[-2], bias])
    #don't forget the bias!
    delta_wo = np.dot(grad.transpose(), hidden_out_bias) * LR_O
    
    #and finally, old weights + delta weights -> new weights!
    w_new.append(weights[-1] - delta_wo.transpose())
    
    # l=2 is the second-last layer
    for l in range(2,len(output)):
        '''EQUATION D:'''
        sig_deriv = output[-l] * ( 1 - output[-l])
        w = weights[-l+1]
        #exclude the bias (3rd column) of the outer weights,
        #since it is not backpropagated!
        ## ??
        #grad = np.dot(grad,w[:-1].transpose())
        grad = sig_deriv * np.dot(grad, w[:-1].transpose())
        '''EQUATION E:'''
        bias =  np.ones((output[-l-1].shape[0], 1))
        z = np.hstack([output[-l-1],bias])
        delta_wH = np.dot(grad.transpose(), z)*LR_H
        w_new.append(weights[-l] - delta_wH.transpose())

    return list(reversed(w_new))

def add_bias(X):
    return np.hstack((X, np.ones((X.shape[0], 1))))

def artificial_neural_network(X,y,neurons_per_layer,
                              epochs=500,
                              batch_size=25,
                              LR_H=0.01,
                              LR_O=0.01):
    '''
    input_per_layer -- list of int, length of list is the number of inputs per layer, and the ith element of the list is the number of neurons in the ith hidden layer. The output layer always has exactly one neuron.
    '''
    inp = X.shape[1] + 1 # add bias
    weights = []
    for neurons in neurons_per_layer:
        weights.append(np.random.normal(size=(inp, neurons)))
        # number of inputs of the next layer equals number of neurons of the previous layer + 1
        inp = neurons + 1
    weights.append(np.random.normal(size=(inp, 1)))

    LOSS_VEC = []
    ACC_VEC = []
    for _ in range(epochs):
        for Xbatch,ybatch in batch(X,y,batch_size):
            X_bias =add_bias(Xbatch)
            out = feed_forward(X_bias, weights)
            weights = backprop(weights,out,ybatch,Xbatch,LR_H,LR_O)
        loss = sum(log_loss(ybatch.reshape(-1,1), out[-1]))[0]
        LOSS_VEC.append(loss)
        ypred=out[-1].round()
        acc = sum((ypred) == ybatch.reshape(-1,1)) / len(ybatch)
        ACC_VEC.append(acc)
    return weights, LOSS_VEC, ACC_VEC

def batch(X,y,n):
    '''Yields chunks of size n of the dataset'''
    for i in range(0,len(X),n):
        yield X[i:i+n], y[i:i+n]

def visualize(LOSS_VEC, ACC_VEC):
    plt.figure(figsize =(12,6))
    plt.subplot(121)
    plt.plot(ACC_VEC)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(122)
    plt.plot(LOSS_VEC)
    plt.title('model loss')
    plt.ylabel('log loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='lower right')

    plt.show()


def main():
    # Get dataset 
    X,y = make_moons(n_samples=50, noise=0.2, random_state=43)

    ## ANN with two layers, first layer has two nodes and the output layer has one node
    w, LOSS_VEC, ACC_VEC  = artificial_neural_network(X,y,
                        [2],
                        epochs=5000,
                        batch_size=25)
    print(f"Maximum Loss : {max(LOSS_VEC)}")
    print("")
    print(f"Minimum Loss : {min(LOSS_VEC)}")
    print("")
    print(f"Final Accuracy : {ACC_VEC[-1]}")
    print("")
    #plt.title("ANN with 2 layers")
    visualize(LOSS_VEC, ACC_VEC)

    ## ANN with five layers
    w, LOSS_VEC, ACC_VEC  = artificial_neural_network(
                        X,
                        y,
                        [4,6,6,4],
                        epochs=5000,
                        batch_size=25,
                        LR_H=0.03,
                        LR_O=0.03)
    #plt.title("ANN with 5 layers")
    visualize(LOSS_VEC, ACC_VEC)
    print(f"Maximum Loss : {max(LOSS_VEC)}")
    print("")
    print(f"Minimum Loss : {min(LOSS_VEC)}")
    print("")
    print(f"Final Accuracy : {ACC_VEC[-1]}")
    print("")

    # Create a grid in the rectangle $-2<x,y<3$ 
    xx = np.linspace(-2, 3, 40)
    yy = np.linspace(-2, 3, 40)
    gx, gy = np.meshgrid(xx, yy)

    Z = np.c_[gx.ravel(), gy.ravel()]
    out = feed_forward(add_bias(Z), w)
    ypred=out[-1] #.round()
    ypred =ypred.reshape(gx.shape)
    plt.contourf(gx, gy, ypred, cmap=plt.cm.coolwarm, alpha=0.8)

    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-2, 3])

    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Model predictions on our Training set')

    plt.show()


if __name__ == '__main__':
    main()

