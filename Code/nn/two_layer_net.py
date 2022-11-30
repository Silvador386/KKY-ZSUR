from Code.nn.layer_utils import *
from Code.nn.layers import softmax_loss


class TwoLayerNet:
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design.
    """

    def __init__(
        self,
        input_dim=1 * 2,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        # Initialize parameters
        self.params["W1"] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params["W2"] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b1"] = np.zeros(shape=hidden_dim)
        self.params["b2"] = np.zeros(shape=num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        # Forward pass
        out1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2 = affine_forward(out1, self.params["W2"], self.params["b2"])
        scores = out2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*self.reg*(np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))

        # Backpropagation
        dx2, grads["W2"], grads["b2"] = affine_backward(dscores, cache2)
        dx1, grads["W1"], grads["b1"] = affine_relu_backward(dx2, cache1)

        grads["W2"] += self.reg * self.params["W2"]
        grads["W1"] += self.reg * self.params["W1"]

        return loss, grads
