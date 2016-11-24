
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
from mlp.models import MultipleLayerModel
from mlp.layers import ReluLayer, AffineLayer, DropoutLayer, ReshapeLayer, ConvolutionalLayer, MaxPooling2DLayer, BatchNormalization
from mlp.errors import CrossEntropySoftmaxError
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import MomentumLearningRule
from mlp.optimisers import Optimiser
import matplotlib.pyplot as plt


def show_data(train_data):

    cnt = 0
    xshape = None
    yshape = None
    for x, y in train_data:
        cnt += x.shape[0]
        xshape = x.shape
        yshape = y.shape

    return cnt, xshape, yshape


def test_bn(params):
    # Seed a random number generator
    seed = params['seed']
    rng = np.random.RandomState(seed)

    # Set up a logger object to print info about the training run to stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler()]

    # Create data provider objects for the MNIST data set
    train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
    valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

    train_size, xshape, yshape = show_data(train_data)
    valid_size, _, _ = show_data(valid_data)
    print 'train size %d, valid size %d' % (train_size, valid_size)
    print xshape
    print yshape

    print 'load data done!'

    # Probability of input being included in output in dropout layer
    incl_prob = 0.8

    input_dim, output_dim, hidden_dim = 784, 10, 125

    # Use Glorot initialisation scheme for weights and zero biases
    weights_init = GlorotUniformInit(rng=rng, gain=2.**0.5)
    biases_init = ConstantInit(0.)

    # Create three affine layer model with rectified linear non-linearities
    # and dropout layers before every affine layer
    model = MultipleLayerModel([
        DropoutLayer(rng, incl_prob),
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        BatchNormalization(hidden_dim),
        ReluLayer(),
        DropoutLayer(rng, incl_prob),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
        BatchNormalization(hidden_dim),
        ReluLayer(),
        DropoutLayer(rng, incl_prob),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])


    # Multiclass classification therefore use cross-entropy + softmax error
    error = CrossEntropySoftmaxError()

    # Use a momentum learning rule - you could use an adaptive learning rule
    # implemented for the coursework here instead
    learning_rule = MomentumLearningRule(0.02, 0.9)

    # Monitor classification accuracy during training
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    optimiser = Optimiser(
        model, error, learning_rule, train_data, 
        valid_data, data_monitors, use_stochastic_eval = False)

    num_epochs = 100
    stats_interval = 5

    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')


if __name__ == '__main__':
    params = dict(seed = 123)
    test_bn(params)


