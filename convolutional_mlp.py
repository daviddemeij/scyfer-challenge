"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation=T.tanh):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

# To make the code more easily adjustable and testable I transformed the original evaluate_lenet5 function into a class
class lenet5:
    def __init__(self,
                        dataset='mnist.pkl.gz',
                        nkerns=[20, 50], batch_size=500, update_rule = 'regular', config=None, dropout=0, activation='tanh'):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        if activation=='tanh':
            activation_fn = T.tanh
        # Set activation function to none because with PreLU additional alpha variables have to be initialized
        # by setting the activation function to None the linear activation will be retrieved which then can be
        # activated by my PreLU implementation
        elif activation=='PreLU':
            activation_fn = None

        rng = numpy.random.RandomState(23455)

        datasets = load_data(dataset)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        self.n_train_batches = n_train_batches
        self.n_valid_batches = n_valid_batches
        self.n_test_batches = n_test_batches
        self.loss_history = []
        self.val_error_history = []
        self.train_error_history = []
        # allocate symbolic variables for the data

        index = T.lscalar()  # index to a [mini]batch
        mode = T.lscalar() # 1 = training (dropout enabled), 0 = testing (dropout disabled)

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2),
            activation = activation_fn
        )
        if(activation=='PreLU'):
            ########################
            # PreLU Implementation #
            ########################
            # if the activation function is PreLU alpha has to be initialized with the same shape as the bias
            # alpha will be initialized at 0.25 as suggested in the article that introduced PreLU
            # Reference: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
            # (Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun, Microsoft, 2015)
            alpha0 = theano.shared(numpy.ones(layer0.b.get_value().shape,dtype=theano.config.floatX)*0.25, borrow=True)
            layer1_input = self.PreLU(layer0.output, alpha0.dimshuffle('x', 0, 'x', 'x'))
        else:
            layer1_input = layer0.output

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer1_input,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2),
            activation = activation_fn
        )
        if (activation == 'PreLU'):
            alpha1 = theano.shared(numpy.ones(layer1.b.get_value().shape, dtype=theano.config.floatX) * 0.25,
                                   borrow=True)
            layer1_output = self.PreLU(layer1.output, alpha1.dimshuffle('x', 0, 'x', 'x'))
        else:
            layer1_output = layer1.output

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1_output.flatten(2)

        # Add dropout if dropout value is higher than 0 and in training mode
        if(dropout>0):
            layer2_input = theano.ifelse.ifelse(theano.tensor.eq(mode, 1), self.Dropout(layer2_input, dropout, rng), layer2_input)

        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=activation_fn
        )
        if (activation == 'PreLU'):
            alpha2 = theano.shared(numpy.ones(layer2.b.get_value().shape, dtype=theano.config.floatX) * 0.25,
                                   borrow=True)
            layer2_output = self.PreLU(layer2.output, alpha2)
        else:
            layer2_output = layer2.output

        # Add dropout if dropout value is higher than 0 and in training mode
        if (dropout > 0):
            layer3_input = theano.ifelse.ifelse(theano.tensor.eq(mode, 1), self.Dropout(layer2_output, dropout, rng),
                                                layer2_output)
        else:
            layer3_input = layer2_output

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer3_input, n_in=500, n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)
        #self.print_output = theano.function(
        #    [index],
        #    [alpha0.dimshuffle('x',0,'x','x'), layer0.b, layer0.output],
        #    givens={
        #        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        #    },
        #    on_unused_input='ignore'
        #)
        self.print_layer2 = theano.function(
            [index],
            layer2_input,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                mode: 1
            },
            on_unused_input = 'ignore' # if dropout<0 the 'mode' variable will be unused
        )
        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                mode: 0
            },
            on_unused_input = 'ignore' # if dropout<0 the 'mode' variable will be unused
        )

        self.validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                mode: 0
            },
            on_unused_input = 'ignore' # if dropout<0 the 'mode' variable will be unused
        )
        self.train_error_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                mode: 0
            },
            on_unused_input = 'ignore' # if dropout<0 the 'mode' variable will be unused
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer3.params + layer2.params + layer1.params + layer0.params

        if activation == 'PreLU':
            alpha = [alpha0, alpha1, alpha2]
            params += alpha
        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (paras[i], grads[i]) pairs.
        if update_rule=='regular':
            if(config is None) : config = {}
            config.setdefault('learning_rate', 0.1)
            updates = [
                (param, param - 0.1 * grad)
                for param, grad in zip(params, grads)
            ]
            ###########################
            # AdaDelta implementation #
            ###########################
            # Implementing the adaDelta update rule as described in AdaDelta: An adaptive learning rate method
            # (Matthew D. Zeiler, Google, 2012)
        elif update_rule=='adaDelta':

            if(config is None): config = {}
            config.setdefault('decay_rate',0.95)
            config.setdefault('epsilon',1e-6)
            config.setdefault('learning_rate', 1.)
            
            # E(g^2) is a Theano variable to store the moving average of the squared gradient
            Egrads = [
                theano.shared(numpy.zeros_like(param.get_value(),dtype=theano.config.floatX),borrow=True)
                for param in params
                ]
            # E(dx^2) is a Theano variable to store the moving average of the squared updates to the parameters
            Edxs = [
                theano.shared(numpy.zeros_like(param.get_value(),dtype=theano.config.floatX),borrow=True)
                for param in params
            ]
            # The updated E(g^2) value is calculated and will be added to the parameter updates
            Egrads_new = [
                config['decay_rate'] * Egrad + (1 - config['decay_rate']) * (grad ** 2)
                for (Egrad, grad) in zip(Egrads, grads)
            ]
            # The parameter update is calculated using the AdaDelta update rule
            dxs = [
                -(T.sqrt(Edx + config['epsilon']) / T.sqrt(Egrad_new + config['epsilon'])) * grad
                for (Edx, Egrad_new, grad) in zip(Edxs, Egrads_new, grads)
                ]
            # The updated E(dx^2) value is calculated and will be added to the parameter updates
            Edxs_new = [
                config['decay_rate']*Edx + (1-config['decay_rate']) * (dx ** 2)
                for (Edx, dx) in zip(Edxs, dxs)
            ]
            Egrads_updates = zip(Egrads, Egrads_new)
            Edxs_updates = zip(Edxs, Edxs_new)
            param_updates = [
                (param, param+dx)
                for (param, dx) in zip(params, dxs)
            ]
            # The new E(g^2) and E(dx^2) are added to the parameter updates so they will be updated at the same time
            # as the model parameters.
            updates = param_updates + Egrads_updates + Edxs_updates

        else:
            raise ValueError('Unrecognized update rule %s' % update_rule)
        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                mode: 1 # in training mode dropout is enabled (if dropout>0)
            },
            on_unused_input = 'ignore' # if dropout<0 the 'mode' variable will be unused
        )

        # end-snippet-1

        ###############
        # TRAIN MODEL #
        ###############

    def train(self, n_epochs=10):
        print('... training')
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_error = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                self.loss_history.append(self.train_model(minibatch_index))

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_errors = [self.validate_model(i) for i
                                         in range(self.n_valid_batches)]
                    this_validation_error = numpy.mean(validation_errors)
                    self.val_error_history.append(this_validation_error)


                    # Also compute training error, to check for overfitting
                    training_errors = [self.train_error_model(i) for i
                                         in range(self.n_train_batches)]
                    this_train_error = numpy.mean(training_errors)
                    print('epoch %i, minibatch %i/%i, training error %f %%, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_train_error * 100., this_validation_error * 100.))
                    self.train_error_history.append(this_train_error)

                    # if we got the best validation score until now
                    if this_validation_error < best_validation_error:

                        #improve patience if loss improvement is good enough
                        if this_validation_error < best_validation_error *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_error = this_validation_error
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_error * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def Dropout(self, input, dropout, rng):
        ##########################
        # Dropout implementation #
        ##########################
        # Dropout was introduced in Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Nitish Srivastava;
        # Geoffrey Hinton; Alex Krizhevsky; Ilya Sutskever; Ruslan Salakhutdinov, 2014) and aims to reduce overfitting
        # by randomly dropping out a certain portion of the hidden units.

        # Create a matrix of the same size as the input where for each value in the matrix the value will be set to 1 with
        # a probability of (1-dropout) and the value will be set to zero with a probability of dropout
        mask = T.shared_randomstreams.RandomStreams(rng.randint(123456)).binomial(n=1, p=1 - dropout, size=input.shape)

        # multiply the input with the mask so a certain part of the input is dropped and scale the output so there is
        # no need to scale the inputs at test mode
        return input * T.cast(mask, theano.config.floatX) * (1./(1-dropout))

    def PreLU(self, input, alpha):
        return T.switch(input < 0, alpha * input, input)



