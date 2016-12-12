#!/usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
from scipy.sparse import lil_matrix
from stochastic_bb import svrg_bb, sgd_bb


"""
    An example showing how to use svrg_bb and sgd_bb
    The problem here is the regularized logistic regression
"""

__license__ = 'MIT'
__author__ = 'Conghui Tan'
__email__ = 'tanconghui@gmail.com'


if __name__ == '__main__':
    # problem size
    n, d = 1000, 100

    # randomly generate data
    A = np.random.randn(n, d)
    x_true = np.random.randn(d)
    y = np.sign(np.dot(A, x_true) + 0.1 * np.random.randn(n))

    # generate test data
    A_test = np.random.randn(n, d)
    y_test = np.sign(np.dot(A_test, x_true))

    # define function and gradient via Theano
    tmp = lil_matrix((n, n))
    tmp.setdiag(y)
    data = theano.shared(tmp * A)

    l2 = 1e-2
    par = T.vector()
    loss = T.log(1 + T.exp(-T.dot(data, par))).mean() + l2 / 2 * (par ** 2).sum()
    func = theano.function(inputs=[par], outputs=loss)

    idx = T.ivector()
    grad = theano.function(inputs=[par, idx], outputs=T.grad(loss, wrt=par),
                           givens={data: data[idx, :]})

    x0 = np.random.rand(d)
    print('Begin to run SVRG-BB:')
    x = svrg_bb(grad, 1e-3, n, d, func=func, max_epoch=50)
    y_pred = np.sign(np.dot(A_test, x))
    print('Test accuracy: %f' % (np.count_nonzero(y_test == y_pred) / n))

    print('\nBegin to run SGD-BB:')
    x = sgd_bb(grad, 1e-3, n, d, phi=lambda k: k, func=func, max_epoch=50)
    y_pred = np.sign(np.dot(A_test, x))
    print('Test accuracy: %f' % (np.count_nonzero(y_test == y_pred) / n))
