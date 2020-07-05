# feedforward
Python implementation of a simple artificial neural network (ANN)

The information in the network moves only in one direction, from the input layer through the hidden layer and to the output layer. Hence, it is a Feedforward neural network.
Each neuron in the current layer is connected to each neuron in the subsequent layer, it is a fully connected net.

The function `feed_forward()` gives the output of the neural net for a given input `X`and weights `weights`. `weights` must be a list of the matrices (numpy arrays) $W^l$ for each layer $l$. If $W^l$ is a $n\cross m$-matrix, then $n-1$ is the number of inputs and $m$ the number of neurons in the $l$th layer. Here, the bias is considered as an additional input, that is always $1$.
Let $A_{l-1}$ be the output of the previous layer with an additional row of ones added (representing the bias). Then, starting with the first neuron, the output is calculated by computing the matrix multiplication $A_{l-1}W^L$ and applying the logistic function `sigmoid()` to each value of the product matrix.
This process is also called forward-probagation.


Backprogation is used to compute the gradient of the loss function over the space of all possible weigths.
This process is implemented in the function `backprop()` for a dense neural net, where the output layer consists of 1 neuron and the loss function is the log loss. This is a binary classifier.

In general, the goal of any supervised learning algorithm is to find a functions that maps the set of inputs $X$ to their correct output $y$.
The function `artificial_neural_network()` takes as input the training data $X$ with desired output $y$ and the hyperparameters number of `epochs`, batch_size and the learning rate `LR_H` for the hidden and `LR_O` for the outer layer.
The weights $W$ are first initialised randomly and the predicted output of the neural net is computed with `feed_forward()`.
The loss function `log_loss()` is used to calculate the discrepency between the predicted and the actual outputs $y$

## ToDo:
- Allow the use of different activation functions, like ReLu or tanh
- Allow a different kind of output layer
- Implement different kinds of loss functions