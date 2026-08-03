"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np

def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    return np.argmax(matrix, axis=1)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
    return np.max(matrix, axis=1, keepdims= True)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    return np.sum(matrix, axis=1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    row_max = np.max(logits, axis=1, keepdims=True)
    return np.exp(logits - row_max )

# Step 5 - stable_softmax
import numpy as np
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    row_max = np.max(logits, axis=1, keepdims=True)
    logits_shifted = logits - row_max
    exp_shifted = np.exp(logits_shifted)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    return exp_shifted / sum_exp

# Step 6 - one_hot
def one_hot(labels, num_classes):
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    num_samples = labels.shape[0]
    out = np.zeros((num_samples, num_classes), dtype=np.float32)
    out[np.arange(num_samples),labels] = 1.0
    return out

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    n_samples = probs.shape[0]
    return probs[np.arange(n_samples), labels]

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    true_class_probs = gather_true_class_probs(probs , labels)
    clipped_probs = np.clip(true_class_probs, eps, None)
    return -np.mean(np.log(clipped_probs))

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    predictions = np.argmax(logits_or_probs, axis=1)
    return np.mean(predictions == labels)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return (2 / fan_in)** 0.5

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    std = he_std((fan_in))
    return np.random.normal(loc=0.0, scale=std, size=shape).astype(np.float64)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length, dtype=np.float64)

# Step 13 - pad_2d
def pad_2d(images, pad):
    # TODO: zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.
    pad_width = ((0, 0), (0,0),(pad, pad), (pad, pad))
    return np.pad(images, pad_width, mode='constant', constant_values=0)

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    
    return (input_size - kernel + 2 * padding) // stride + 1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
   
    N, C, H, W = images.shape

    out_h = (H - kernel_h + 2 * padding) // stride + 1
    out_w = (W - kernel_w + 2 * padding) // stride + 1
    
    images_padded = np.pad(
        images, 
        ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
        mode='constant'
    )
    
    cols = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=images.dtype)
    
    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            cols[:, :, y, x, :, :] = images_padded[:, :, y:y_max:stride, x:x_max:stride]
            
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * kernel_h * kernel_w)

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    N, C, H, W = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols_reshaped = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)
    cols_reshaped = cols_reshaped.transpose(0, 3, 4, 5, 1, 2)

    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    images_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            images_padded[:, :, y:y_max:stride, x:x_max:stride] += cols_reshaped[:, :, y, x, :, :]

    if padding == 0:
        return images_padded

    return images_padded[:, :, padding:-padding, padding:-padding]

# Step 17 - conv2d_forward
def conv2d_forward(images, weights, bias, stride=1, padding=0):
    N, C, H, W = images.shape
    F, Cw, kernel_h, kernel_w = weights.shape

    cols = im2col(images, kernel_h, kernel_w, stride, padding)
    weights_col = weights.reshape(F, -1).T

    out = cols @ weights_col + bias
    
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)


    out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    cache = {
        "images": images,
        "weights": weights,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "cols": cols,
        "out_h": out_h,
        "out_w": out_w,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w
    }

    return out, cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    weights = cache["weights"]
    images = cache["images"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]

    N, C, H, W = images.shape
    F, _, _, _ = weights.shape

    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, F)

    weights_col = weights.reshape(F, -1)
    dcols = d_out_reshaped @ weights_col
    dx = col2im(dcols, images.shape, kernel_h, kernel_w, stride, padding)

    return dx

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    cols = cache["cols"]
    weights = cache["weights"]

    F, C_in, kH, kW = weights.shape
    d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, F)
    dW_col = cols.T @ d_out_reshaped
    dW = dW_col.T.reshape(F, C_in, kH, kW)

    return dW

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    db = np.sum(d_out, axis=(0, 2, 3))
    
    return db

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    dx = conv2d_grad_input(d_out, cache)
    dW = conv2d_grad_weights(d_out, cache)
    db = conv2d_grad_bias(d_out)
    
    return dx, dW, db

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    N, C, H, W = x.shape

    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)

    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=np.int64)

    for i in range(out_h):
        h_start = i * stride
        h_end = h_start + kernel
        for j in range(out_w):
            w_start = j * stride
            w_end = w_start + kernel

            # window: (N, C, kernel, kernel)
            window = x[:, :, h_start:h_end, w_start:w_end]

            # flatten the window -> (N, C, kernel*kernel)
            window_flat = window.reshape(N, C, -1)

            # max value and flat in-window index
            idx = np.argmax(window_flat, axis=2)
            out[:, :, i, j] = np.max(window_flat, axis=2)
            argmax[:, :, i, j] = idx

    cache = {
        "x_shape": x.shape,
        "argmax": argmax,
        "kernel": kernel,
        "stride": stride,
    }

    return out, cache

# Step 23 - scatter_grad_window
def scatter_grad_window(grad_value, argmax_index, kernel):
    grad_window = np.zeros(kernel * kernel)
    grad_window[argmax_index] = grad_value
    return grad_window.reshape(kernel, kernel)

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    x_shape = cache["x_shape"]
    argmax = cache["argmax"]
    kernel = cache["kernel"]
    stride = cache["stride"]

    N, C, H, W = x_shape
    _, _, out_h, out_w = d_out.shape

    dx = np.zeros(x_shape, dtype=d_out.dtype)

    for i in range(out_h):
        h_start = i * stride
        for j in range(out_w):
            w_start = j * stride

            idx = argmax[:, :, i, j]          # (N, C)
            rows = h_start + idx // kernel
            cols = w_start + idx % kernel

            n_idx = np.arange(N)[:, None]
            c_idx = np.arange(C)[None, :]

            np.add.at(dx, (n_idx, c_idx, rows, cols), d_out[:, :, i, j])

    return dx

# Step 25 - relu_forward
def relu_forward(x):
    out = np.maximum(0, x)
    cache = {"x": x}

    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    x = cache['x']
    mask = x > 0
    dx = d_out * mask
    
    return dx

# Step 27 - flatten_forward
def flatten_forward(x):
    cache = {'x_shape': x.shape}
    N = x.shape[0]
    out = x.reshape(N, -1)
    
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    x_shape = cache['x_shape']
    dx = d_out.reshape(x_shape)
    
    return dx

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    out = x @ weights + bias
    cache = {
        'x': x,
        'weights': weights
    }
    
    return out, cache

# Step 30 - linear_grad_input
def linear_grad_input(d_out, cache):
    weights = cache['weights']
    dx = d_out @ weights.T
    
    return dx

# Step 31 - linear_grad_weights
def linear_grad_weights(x, dout):
    dW = x.T @ dout
    
    return dW

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    db = np.sum(dout, axis=0)
    
    return db

# Step 33 - linear_backward
def linear_backward(dout, cache):
    # TODO: combine input, weight, and bias gradients for a linear layer using the cache
    x = cache['x']
    weights = cache['weights']
    dx = dout @ weights.T          
    dW = x.T @ dout                
    db = dout.sum(axis=0)          
    
    return dx, dW, db

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    probs = stable_softmax(logits)
    loss = cross_entropy_loss(probs, y)
    return float(abs(loss))

# Step 35 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(logits, y):
    
    N = logits.shape[0]
    probs = stable_softmax(logits)
    grads = probs.copy()
    grads[np.arange(N), y] -= 1
    
    grads /= N
    
    return grads

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    # Standard SGD update
    return param - lr * grad

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    return beta_one * m + (1 - beta_one) * grad

# Step 38 - adam_update_v
import numpy as np

def adam_update_v(v, grad, beta_two):
    return beta_two * v + (1 - beta_two) * (grad ** 2)

# Step 39 - adam_bias_correct
def adam_bias_correct(moment, beta, t):
    # TODO: return moment divided by (1 - beta**t) to undo Adam's zero-init bias.
    return moment / (1 - beta**t)

# Step 40 - adam_param_step
def adam_param_step(param, m_hat, v_hat, lr, eps):
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    new_m = adam_update_m(m, grad, beta_one)
    new_v = adam_update_v(v, grad, beta_two)

    m_hat = adam_bias_correct(new_m, beta_one, t)
    v_hat = adam_bias_correct(new_v, beta_two, t)

    new_param = adam_param_step(param, m_hat, v_hat, lr, eps)
    
    return new_param, new_m, new_v

# Step 42 - init_conv_layer
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    fan_in = in_channels * kernel_size * kernel_size
    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)

    W = he_init(weight_shape, fan_in, seed)
    b = init_zero_bias(out_channels)

    return {'W': W, 'b': b}

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    fan_in = in_features
    weight_shape = (in_features, out_features)

    W = he_init(weight_shape, fan_in, seed)
    b = init_zero_bias(out_features)

    return {'W': W, 'b': b}

# Step 44 - init_lenet
def init_lenet(in_channels, num_classes, seed=0):
    params = {}

    # Convolution layers
    params['conv1'] = init_conv_layer(
        out_channels=6,
        in_channels=in_channels,
        kernel_size=5,
        seed=seed
    )

    params['conv2'] = init_conv_layer(
        out_channels=16,
        in_channels=6,
        kernel_size=5,
        seed=seed + 1
    )

    # Fully connected layers
    params['fc1'] = init_linear_layer(
        in_features=16 * 4 * 4,
        out_features=120,
        seed=seed + 2
    )

    params['fc2'] = init_linear_layer(
        in_features=120,
        out_features=num_classes,
        seed=seed + 3
    )

    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, W, b, pool_size, stride, padding):
    # Convolution
    conv_out, conv_cache = conv2d_forward(x, W, b, stride, padding)

    # ReLU activation
    relu_out, relu_cache = relu_forward(conv_out)

    # Non-overlapping max pooling (pool stride == pool window)
    pool_out, pool_cache = maxpool2d_forward(relu_out, pool_size, pool_size)

    cache = {
        'conv_cache': conv_cache,
        'relu_cache': relu_cache,
        'pool_cache': pool_cache,
    }

    return pool_out, cache

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
   
    flat_out, flatten_cache = flatten_forward(x)

    fc1_out, fc1_cache = linear_forward(flat_out, fc1['W'], fc1['b'])

    relu_out, relu_cache = relu_forward(fc1_out)

    logits, fc2_cache = linear_forward(relu_out, fc2['W'], fc2['b'])

    cache = {
        'flatten_cache': flatten_cache,
        'fc1_cache': fc1_cache,
        'relu_cache': relu_cache,
        'fc2_cache': fc2_cache,
    }

    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    # Conv block 1: conv (stride=1, pad=0) -> ReLU -> 2x2 pool
    out1, cache1 = forward_conv_block(
        x,
        params['conv1']['W'],
        params['conv1']['b'],
        pool_size=2,
        stride=1,
        padding=0,
    )

    # Conv block 2
    out2, cache2 = forward_conv_block(
        out1,
        params['conv2']['W'],
        params['conv2']['b'],
        pool_size=2,
        stride=1,
        padding=0,
    )

    # Dense classifier head
    logits, cache_cls = forward_classifier_block(
        out2,
        params['fc1'],
        params['fc2'],
    )

    caches = {
        'block1': cache1,
        'block2': cache2,
        'classifier': cache_cls,
    }
    return logits, caches

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
    # Unpack caches from forward_conv_block
    conv_cache = cache['conv_cache']
    relu_cache = cache['relu_cache']
    pool_cache = cache['pool_cache']

    # Reverse order: pool -> relu -> conv
    dpool = maxpool2d_backward(dout, pool_cache)
    drelu = relu_backward(dpool, relu_cache)
    dx, dW, db = conv2d_backward(drelu, conv_cache)

    return dx, dW, db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    # Unpack caches
    flatten_cache = cache['flatten_cache']
    fc1_cache = cache['fc1_cache']
    relu_cache = cache['relu_cache']
    fc2_cache = cache['fc2_cache']

    drelu, dW2, db2 = linear_backward(dlogits, fc2_cache)

    # Backprop through ReLU
    dfc1 = relu_backward(drelu, relu_cache)

    # Backprop through fc1
    dflat, dW1, db1 = linear_backward(dfc1, fc1_cache)

    # Backprop through flatten
    dx = flatten_backward(dflat, flatten_cache)

    return {
        'dx': dx,
        'fc1': {'dW': dW1, 'db': db1},
        'fc2': {'dW': dW2, 'db': db2},
    }

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

