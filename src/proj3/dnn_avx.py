import os
import sys
import math
import networkx as nx
import numpy as np
import ctypes
from ctypes import *
mylib = cdll.LoadLibrary('./avx_lib.so')

'''
    Reference: CS231n assignment 2 im2col
'''
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H +  padding - field_height) % stride == 0
    # assert (W + padding - field_height) % stride == 0
    out_height = (H + padding - field_height) / stride + 1
    out_width = (W + padding - field_width) / stride + 1
    out_height = int(out_height)
    out_width = int(out_width)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p//2, (p+1)//2), (p//2, (p+1)//2)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

class DnnInferenceEngine(object):
    def __init__(self, graph, debug):
        self.g = graph
        self.debug = debug

    def run(self, tin):
        print("-------------")
        print("Using AVX")
        print("-------------")
        self.g.in_node.set_input(tin)
        out = {}
        currents = [self.g.in_node]
        done = set()
        # i = 0
        while (len(currents) != 0):
            nexts = []
            for current in currents:
                skip_current = False
                predecessors = self.g.G.predecessors(current)
                for predecessor in predecessors:
                    if predecessor not in done:
                        nexts.append(predecessor)
                        skip_current = True
                if skip_current:
                    continue
                current.run()
                if self.debug:
                    np.save("../../intermediate/{}.npy".format(current.name), current.result)
                # if i != 0:
                #     tf_current = np.load("../../YoloTinyV2/intermediate/layer_{}.npy".format(i))
                #     print("Layer{}: ".format(i),np.sum(np.absolute(tf_current - current.result)))
                # i+=1
                if self.g.is_out_node(current):
                    out = current.result
                done.add(current)
                for successor in self.g.G.successors(current):
                    nexts.append(successor)
            currents = nexts[:]
        return out

class DnnGraphBuilder(object):
    def __init__(self):
        self.G = nx.DiGraph()
        self.name_num = {"conv2d": 0, 
                         "bias_add": 0, 
                         "max_pool2d": 0, 
                         "batch_norm": 0, 
                         "leaky_relu": 0, 
                         "input": 0}
        self.in_node = None
        self.out_node = None

    def set_in_node(self, node):
        self.in_node = node

    def set_out_node(self, node):
        self.out_node = node

    def is_out_node(self, node):
        return self.out_node is node

    def get_name(self, layer_name):
        name = layer_name + "_" + str(self.name_num[layer_name])
        self.name_num[layer_name] += 1
        return name

    def create_conv2d(self, in_node, kernel, strides, padding):
        out_node = Conv2D(self.get_name("conv2d"), in_node, kernel, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_bias_add(self, in_node, biases):
        out_node = BiasAdd(self.get_name("bias_add"), in_node, biases)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_max_pool2d(self, in_node, ksize, strides, padding):
        out_node = MaxPool2D(self.get_name("max_pool2d"), in_node, ksize, strides, padding)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_batch_norm(self, in_node, mean, variance, gamma, epsilon):
        out_node = BatchNorm(self.get_name("batch_norm"), in_node, mean, variance, gamma, epsilon)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_leaky_relu(self, in_node):
        out_node = LeakyReLU(self.get_name("leaky_relu"), in_node)
        self.G.add_edge(in_node, out_node)
        return out_node

    def create_input(self, in_shape):
        out_node = Input(self.get_name("input"), in_shape)
        self.G.add_node(out_node) 
        self.set_in_node(out_node)  # Assume there's only one input
        return out_node

class DnnNode(object):
    def __init__(self):
        pass

    def run(self):
        self.result = None 

#
# Complete below classes.
#

class Conv2D(DnnNode):
    def __init__(self, name, in_node, kernel, strides, padding):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        filter_height, filter_width, kernel_in_channels, out_channels = kernel.shape

        assert in_channels == kernel_in_channels, \
        "Shape of filters must be same to number of input channels, %d is not equal to %d" % (kernel_in_channels, in_channels)

        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.name = name
        print(self.name)
        self.kernel = kernel
        self.padding = (padding == "SAME")
        self.strides = strides
        
        self.pad_h = 0
        self.pad_w = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            # to avoid  checking extra cases, we will not divide by two
            self.pad_h = ((self.strides[1] - 1) * in_height + filter_height - self.strides[1])
            self.pad_w = ((self.strides[2] - 1) * in_width + filter_width - self.strides[2])
        output_height  = int(((in_height - filter_height + self.pad_h) / self.strides[1]) + 1)
        output_width   = int(((in_width - filter_width + self.pad_w) / self.strides[2]) + 1)
        # self.prev_res = np.zeros((batch, in_height + self.pad_h, in_width + self.pad_w, in_channels))
        # self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.result = None
        # self.out_shape = self.result.shape
        self.out_shape = (batch, output_height, output_width, out_channels)
        self.filter_height, self.filter_width, self.in_channels, out_channels = kernel.shape

    def run(self):
        h_filter, w_filter, d_filter, n_filters = self.kernel.shape
        W = self.kernel.transpose(3, 2, 0, 1)
        n_x, h_x, w_x, d_x = self.in_node.out_shape
        X = self.in_node.result.transpose(0, 3, 1, 2)
        padding = 0
        if self.padding:
            padding = (self.strides[1] - 1) * h_x + h_filter - self.strides[1]

        h_out = (h_x - h_filter + padding) / self.strides[1] + 1
        w_out = (w_x - w_filter + padding) / self.strides[1] + 1
        h_out, w_out = int(h_out), int(w_out)

        X_col = im2col_indices(X, h_filter, w_filter, padding, stride=self.strides[1])
        W_col = W.reshape(n_filters, -1)

        m, n = W_col.shape
        n1, k = X_col.shape

        assert n1==n, "Shapes do not match"
        X_col_tr = X_col.T
        pad = n%4
        if pad!=0:
            pad = 4 - pad
        W_col_padded = np.pad(W_col, ((0, 0), (0, pad)), 
                                mode='constant', constant_values=0.0)
        X_col_padded = np.pad(X_col_tr, ((0, 0), (0, pad)), 
                                mode='constant', constant_values=0.0)

        A = W_col_padded.astype(c_double)
        A_p = A.ctypes.data_as(POINTER(c_double))

        func = mylib.conv2d
        func.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double),
                            c_size_t, c_size_t, c_size_t]

        B = X_col_padded.astype(c_double)
        C = np.zeros((m, k)).astype(c_double)
        B_p = B.ctypes.data_as(POINTER(c_double))
        C_p = C.ctypes.data_as(POINTER(c_double))
        func(C_p, A_p, B_p, m, n + pad, k)
        out = C.astype("float64")

        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 1, 2, 0)
        self.result = out


class BiasAdd(DnnNode):
    def __init__(self, name, in_node, biases):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        assert in_channels == biases.shape[0], \
        "Shape of biases must be equal to number of input channels, %d is not equal to %d" % (biases.shape[0], in_channels)

        self.biases = biases
        self.name = name
        print(self.name)
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape

    def run(self):
        self.result = np.copy(self.in_node.result)
        batch, output_height, output_width, out_channels = self.out_shape
        res = self.result.reshape(batch*output_height*output_width, out_channels)
        res += self.biases
        self.result = res.reshape(batch, output_height, output_width, out_channels)


class MaxPool2D(DnnNode):
    def __init__(self, name, in_node, ksize, strides, padding):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        out_channels = in_channels
        assert padding in ["SAME", "VALID"], \
        "Invalid padding name: %s, should be either SAME or VALID" % padding

        self.strides = strides
        self.ksize = ksize
        self.name = name
        print(self.name)
        
        self.padding = (padding == "SAME")

        pad_h = 0
        pad_w = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            pad_h = self.ksize[1] - 1
            pad_w = self.ksize[2] - 1
        self.prev_res = np.zeros((batch, in_height + pad_h, in_width + pad_w, in_channels))
        output_height  = int((in_height - self.ksize[1] + pad_h) / self.strides[1] + 1)
        output_width   = int((in_width - self.ksize[2] + pad_w) / self.strides[2] + 1)
        self.result = np.zeros((batch, output_height, output_width, out_channels))
        self.out_shape = self.result.shape
        
    def run(self):
        n, h, w, d = self.in_node.out_shape
        pad = 0
        if self.padding:
            # ((s-1) * x + k -s)/ 2
            pad = self.ksize[1] - 1
        X = self.in_node.result.transpose(0, 3, 1, 2)
        X_reshaped = X.reshape(n * d, 1, h, w)

        X_col = im2col_indices(X_reshaped, self.ksize[1], self.ksize[2], pad, self.strides[1])

        _, h_out, w_out, _ = self.out_shape

        cols = X_col.astype(c_double)
        size, n_1 = X_col.shape
        C = np.zeros((n_1, )).astype(c_double)

        func = mylib.maxpool
        func.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t]
        cols_p = cols.ctypes.data_as(POINTER(c_double))
        C_p = C.ctypes.data_as(POINTER(c_double))
        func(C_p, cols_p, size, n_1)
        out = C.astype("float64")

        # max_idx = np.argmax(X_col, axis=0)
        # out = X_col[max_idx, range(max_idx.size)]

        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(2, 0, 1, 3)
        self.result = out


class BatchNorm(DnnNode):
    def __init__(self, name, in_node, mean, variance, gamma, epsilon):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape

        assert in_channels == mean.shape[0], \
        "Shape of mean must be equal to number of input channels, %d is not equal to %d" % (mean.shape[0], in_channels)

        assert in_channels == variance.shape[0], \
        "Shape of variance must be equal to number of input channels, %d is not equal to %d" % (variance.shape[0], in_channels)

        assert in_channels == gamma.shape[0], \
        "Shape of gamma must be equal to number of input channels, %d is not equal to %d" % (gamma.shape[0], in_channels)
        
        self.name = name
        print(self.name)

        self.mean = mean
        self.epsilon = epsilon
        self.gamma = gamma
        self.variance = variance
        self.result = np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape

    def run(self):
        self.prev_res = self.in_node.result
        batch, output_height, output_width, out_channels = self.out_shape
        std = np.sqrt(self.variance + self.epsilon)
        res = self.result.reshape(batch*output_height*output_width, out_channels)
        prev_res = self.prev_res.reshape(batch*output_height*output_width, out_channels)
        res = self.gamma * (prev_res - self.mean) / std
        self.result = res.reshape(batch, output_height, output_width, out_channels)

class LeakyReLU(DnnNode):
    def __init__(self, name, in_node):
        self.in_node = in_node
        batch, in_height, in_width, in_channels = in_node.out_shape
        self.alpha = 0.1
        self.name = name
        print(self.name)
        self.result=  np.zeros(in_node.out_shape)
        self.out_shape = self.result.shape

    def run(self):
        self.result = np.copy(self.in_node.result)
        self.result[self.result < 0] *= self.alpha


# Do not modify below
class Input(DnnNode):
    def __init__(self, name, in_shape):
        self.name = name
        # print(self.name)
        self.in_shape = in_shape 
        self.out_shape =in_shape
        self.result = np.zeros(self.in_shape)

    def set_input(self, tensor):
        assert tuple(self.in_shape) == tuple(tensor.shape)
        self.result = tensor 

    def run(self):
        pass
