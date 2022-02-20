
# coding: utf-8

#  [1]:本文件仅供学习原理使用, 实际应用中由于速度太慢, 一般使用 Cpython 编写的 fast 版本

# In[1]:


import numpy as np
from deep_networks import *


# In[2]:



def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    
    h_prime = int(1 + (H + 2 * pad - HH) / stride)
    w_prime = int(1 + (W + 2 * pad - WW) / stride)
    result = np.random.rand(N, F, h_prime, w_prime)
    
    x_padding = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    
    for w_index in range(w_prime):
        for h in range(h_prime):
            for f in range(F):
                partial = x_padding[:,:, h * stride: h*stride+HH, w_index * stride: w_index * stride + WW]
                curr_filter = w[f]
                for n in range(N):
                    result[n, f, h, w_index] = np.sum(partial[n] * curr_filter) + b[f]

    cache = (x, w, b, conv_param)
    out = result
    
    return out, cache

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    N, F, h_prime, w_prime = dout.shape

    stride = conv_param['stride']
    pad = conv_param['pad']

    db = np.zeros(b.shape)
    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    
    dx_pad = np.pad(dx, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)

    # compute the gradient
    for n in range(N):
        for f in range(F):
            for h_index in range(h_prime):
                for w_index in range(w_prime):
                    partial_dout = dout[n, f, h_index, w_index]
                    partial_dx = dx_pad[n,:,h_index*stride: h_index* stride + HH, w_index*stride:w_index*stride+WW]
                    partial_dx += w[f] * partial_dout
                    dw[f] += x_pad[n,:,h_index*stride: h_index* stride + HH, w_index*stride:w_index*stride+WW] * partial_dout
                    db[f] += partial_dout
    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


# In[3]:


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    N,C,H,W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    h_prime = int(1 + (H - HH) / stride)
    w_prime = int(1 + (W - WW) / stride)

    result = np.random.rand(N, C, h_prime, w_prime)

    for h_index in range(h_prime):
        for w_index in range(w_prime):
            for c_index in range(C):
                result[:,c_index,h_index, w_index] = np.max(x[:,c_index, h_index*stride:h_index*stride+HH, w_index*stride:w_index*stride+WW], axis=(1,2))
    out = result

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    (x, pool_param) = cache

    N,C,H,W = x.shape
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    h_prime = int(1 + (H - HH) / stride)
    w_prime = int(1 + (W - WW) / stride)

    result = np.random.rand(N, C, h_prime, w_prime)

    dx = np.zeros(x.shape)
    for n in range(N):
        for h_index in range(h_prime):
            for w_index in range(w_prime):
                temp_x = x[n,:,h_index*stride:h_index*stride+HH, w_index*stride:w_index*stride+WW]
                max_x = np.max(temp_x, axis=(1,2)).reshape(C, 1, 1)
                mask = max_x == temp_x
                temp_dout = dout[n, :, h_index, w_index].reshape(C,1,1) * mask
                dx[n,:,h_index*stride:h_index*stride+HH, w_index*stride:w_index*stride+WW] = temp_dout

    return dx


# In[4]:



def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, C, H, W = x.shape
    x_reshape = x.transpose(0,3,2,1).reshape(-1, C)
    out_spacial, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param)
    out = out_spacial.reshape(N,W,H,C).transpose(0,3,2,1)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None


    N, C, H, W = dout.shape
    dx_spacial, dgamma, dbeta = batchnorm_backward(dout.transpose(0,3,2,1).reshape(-1, C), cache)
    dx = dx_spacial.reshape(N, W, H, C).transpose(0,3,2,1)

    return dx, dgamma, dbeta


# In[10]:


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db

####(2)####
def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

####(3)####
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_naive(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_naive(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_naive(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive(da, conv_cache)
    return dx, dw, db


# In[6]:


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['W2'] = np.random.normal(0, weight_scale, (int((H/2)*(W/2)*num_filters), hidden_dim))
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        conv_out_1 , conv_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
        hidden_out, hidden_cache = affine_relu_forward(conv_out_1, self.params['W2'], self.params['b2'])
        scores, out_cache = affine_forward(hidden_out, self.params['W3'], self.params['b3'])

        if y is None:
            return scores

        loss, grads = 0, {}
        
        loss, dout = softmax_loss(scores, y)
        dx_out, grads['W3'], grads['b3'] = affine_backward(dout, out_cache)
        dx_hidden, grads['W2'], grads['b2'] = affine_relu_backward(dx_out, hidden_cache)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx_hidden, conv_cache)

        W1, W2, W3 = (self.params['W1'], self.params['W2'], self.params['W3'])

        loss += self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        grads['W1'] += 2 * self.reg * W1
        grads['W2'] += 2 * self.reg * W2
        grads['W3'] += 2 * self.reg * W3

        return loss, grads


# In[7]:


# model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

