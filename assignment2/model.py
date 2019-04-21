import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        #raise Exception("Not implemented!)"
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input = n_input, n_output = hidden_layer_size))
        self.layers.append(ReLULayer())
        self.layers.append(FullyConnectedLayer(n_input = hidden_layer_size, n_output = n_output))
        

    def compute_loss_and_gradients(self, X, y):
                        
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
       
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        #forward
        X_next = X.copy()
        for layer in self.layers:
            X_next = layer.forward(X_next)
        loss, grad = softmax_with_cross_entropy(X_next, y)
        
        
        #backward
        l2 = 0
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_l2 = 0
            for params in layer.params():
                param = layer.params()[params]
                loss_d, grad_d = l2_regularization(param.value, self.reg)
                param.grad += grad_d
                l2 += loss_d
            grad += grad_l2
        loss +=l2
            

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        for layer in self.layers:
            X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")
        for layer_num in range(len(self.layers)):
            for i in self.layers[layer_num].params():
                result[str(layer_num) + "_" + i] = self.layers[layer_num].params()[i]
        return result
            

        return result
