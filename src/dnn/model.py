from dnn.nn import NeuralNetwork
from dnn.layers import DenseLayer
from dnn.layers.activation import ReLU, Softmax, Dropout

nn = NeuralNetwork()

nn.add_layer(DenseLayer(512))
nn.add_layer(ReLU())
nn.add_layer(Dropout(0.3))
nn.add_layer(DenseLayer(512))
nn.add_layer(ReLU())
nn.add_layer(Dropout(0.2))
nn.add_layer(DenseLayer(7))
nn.add_layer(Softmax())
