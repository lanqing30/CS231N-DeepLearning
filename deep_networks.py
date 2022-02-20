
# coding: utf-8

# In[5]:


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[4]:


from data_utils import load_CIFAR10
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
      mean_image = np.mean(X_train, axis=0)
      X_train -= mean_image
      X_val -= mean_image
      X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }


# In[10]:


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M) D = d_1 * d_2 * d_3 ... d_4
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    x_flatten = x.reshape(x.shape[0], -1) # (N, D)
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    x_flatten = x.reshape(x.shape[0], -1)
    x_shape = x.shape
    N, D = x_flatten.shape
    dx = np.dot(dout, w.T)
    db = np.sum(dout, axis = 0)
    dw = np.dot(x_flatten.T, dout)
    dx = dx.reshape(*x_shape)
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
        - x: Inputs, of any shape
    Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
    """
    out = np.array(x)
    out[out < 0] = 0
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
    Returns:
        - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = np.array(dout)
    dx[x < 0] = 0
    return dx

def affine_relu_forward(x, w, b):
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# In[8]:


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

## Batch Normalization


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    ----------------------------------------------------------------------------
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
      
    out, cache = None, None
    
    if mode == 'train':
        # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_mu = x - mean
        inv_var = 1.0 / np.sqrt(var + eps)
        x_hat = x_mu * inv_var
        out = gamma*x_hat + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        
        cache = x_mu, inv_var, x_hat, gamma 
    
    elif mode == 'test':
        x_normalize = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalize + beta  
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

# In[7]:


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """

    N, D = dout.shape
    x_mu, inv_var, x_hat, gamma = cache

    # intermediate partial derivatives
    dxhat = dout * gamma
    dvar = np.sum((dxhat * x_mu * (-0.5) * (inv_var)**3), axis=0)
    dmu = (np.sum((dxhat * -inv_var), axis=0)) + (dvar * (-2.0 / N) * np.sum(x_mu, axis=0))
    dx1 = dxhat * inv_var
    dx2 = dvar * (2.0 / N) * x_mu
    dx3 = (1.0 / N) * dmu

    # final partial derivatives
    dx = dx1 + dx2 + dx3
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat*dout, axis=0)

    return dx, dgamma, dbeta



def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.
  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.
  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = (np.random.rand(*x.shape) >= p) / (1 - p)
    out = x * mask
  elif mode == 'test':
    out = x

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.
  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  dx = None
  if mode == 'train':
    dx = dout * mask
  elif mode == 'test':
    dx = dout
  return dx




def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params):
    a, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
    out, relu_cache = relu_forward(bn_out)
    cache = [fc_cache, relu_cache, bn_cache]
    return out, cache

def affine_bn_relu_backward(dout, cache):
    fc_cache, relu_cache, bn_cache = cache
    da = relu_backward(dout, relu_cache)
    d_bacth, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
    dx, dw, db = affine_backward(d_bacth, fc_cache)
    return dx, dw, db, dgamma, dbeta



class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    
    layer_input_dim = input_dim
    for i, hd in enumerate(hidden_dims):
        self.params['W%d'%(i+1)] = weight_scale * np.random.randn(layer_input_dim, hd)
        self.params['b%d'%(i+1)] = weight_scale * np.zeros(hd)
        if self.use_batchnorm:
            self.params['gamma%d'%(i+1)] = np.ones(hd)
            self.params['beta%d'%(i+1)] = np.zeros(hd)
        layer_input_dim = hd
    self.params['W%d'%(self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
    self.params['b%d'%(self.num_layers)] = weight_scale * np.zeros(num_classes)
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None
    
    layer_input = X
    ar_cache = {}
    dp_cache = {}
    
    for lay in xrange(self.num_layers-1):
        if self.use_batchnorm:
            layer_input, ar_cache[lay] = affine_bn_relu_forward(layer_input, 
                                        self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)], 
                                        self.params['gamma%d'%(lay+1)], self.params['beta%d'%(lay+1)], self.bn_params[lay])
        else:
            layer_input, ar_cache[lay] = affine_relu_forward(layer_input, self.params['W%d'%(lay+1)], self.params['b%d'%(lay+1)])
            
        if self.use_dropout:
            layer_input,  dp_cache[lay] = dropout_forward(layer_input, self.dropout_param)
            
    ar_out, ar_cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
    scores = ar_out
    

    # If test mode return early
    if mode == 'test':
      return scores

    loss, dscores = softmax_loss(scores, y)
    dhout = dscores
    loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(self.num_layers)] * self.params['W%d'%(self.num_layers)])
    
    dx , dw , db = affine_backward(dhout , ar_cache[self.num_layers])
    grads = {}
    grads['W%d'%(self.num_layers)] = dw + self.reg * self.params['W%d'%(self.num_layers)]
    grads['b%d'%(self.num_layers)] = db
    
    dhout = dx
    for idx in xrange(self.num_layers-1):
        lay = self.num_layers - 1 - idx - 1
        loss = loss + 0.5 * self.reg * np.sum(self.params['W%d'%(lay+1)] * self.params['W%d'%(lay+1)])
        if self.use_dropout:
            dhout = dropout_backward(dhout ,dp_cache[lay])
        if self.use_batchnorm:
            dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhout, ar_cache[lay])
        else:
            dx, dw, db = affine_relu_backward(dhout, ar_cache[lay])
        grads['W%d'%(lay+1)] = dw + self.reg * self.params['W%d'%(lay+1)]
        grads['b%d'%(lay+1)] = db
        if self.use_batchnorm:
            grads['gamma%d'%(lay+1)] = dgamma
            grads['beta%d'%(lay+1)] = dbeta
        dhout = dx

    return loss, grads
