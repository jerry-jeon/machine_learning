from unittest import TestCase

from hw import Perceptrons
from hw import sigmoid
from hw import DeepLearningMachine
import numpy as np
import os

TEST_FILE = os.path.join(os.path.dirname(__file__), 'test_trn.txt')


class TestPercentrons(TestCase):

    def setUp(self):
        self.sample_nodes = [13, 2, 2, 1]
        self.perceptrons = Perceptrons(self.sample_nodes)

    def test_sigmoid_range_0_to_1(self):
        fake_data = np.mat([5.332, 1.233, 8.66]).T
        fake_weight = self.perceptrons.beginning_weight(3, 1)
        assert 0 < sigmoid(fake_weight, fake_data) < 1

    def test_initialize_layer_number(self):
        assert len(self.perceptrons.layers) == len(self.sample_nodes)

    def test_initialize_layer_shape(self):
        check = True
        for index, node_length in enumerate(self.sample_nodes):
            check &= ((node_length, 1) == self.perceptrons.layer(index).shape)

        assert check

    def test_change_layer_augmented_layer_length_same_as_node_length_plus_one(self):
        check = True
        for index, node_length in enumerate(self.sample_nodes):
            self.perceptrons.change_layer(index, [1.0] * node_length)
            check &= ((node_length + 1, 1) == self.perceptrons.augmented_layers[index].shape)

        assert check


    def test_initialize_weight_number(self):
        assert len(self.perceptrons.weights) == len(self.sample_nodes) - 1

    def test_initialize_weight_shape(self):
        check = True

        for i in range(len(self.sample_nodes) - 1):
            check &= ((self.sample_nodes[i], self.sample_nodes[i+1]) == self.perceptrons.weight(i).shape)

        assert check

    def test_initialize_weight_range(self):
        check = True

        for weight in self.perceptrons.weights:
            row, col = weight.shape
            for i in range(row):
                for j in range(col):
                    value = weight.item(i, j)
                    check &= (value >= -0.01) & (value <= 0.01) & (value != 0)

        assert check

    def test_layer_row_vector_matix_return_column_vector(self):
        self.perceptrons.layers[0] = np.mat([1.0, 2.0])
        assert (2, 1) == self.perceptrons.layer(0).shape

    def test_layer_column_vector_matrix_return_column_vector(self):
        self.perceptrons.layers[0] = np.mat([1.0, 2.0]).T
        assert (2, 1) == self.perceptrons.layer(0).shape

    def test_layer_ndarray_return_column_vector(self):
        self.perceptrons.layers[0] = np.array([1.0, 2.0])
        assert (2, 1) == self.perceptrons.layer(0).shape

    def test_layer_list_return_column_vector(self):
        self.perceptrons.layers[0] = [1.0, 2.0]
        assert (2, 1) == self.perceptrons.layer(0).shape

    def test_last_layer(self):
        assert self.perceptrons.last_layer() == self.perceptrons.layers[len(self.perceptrons.layers) - 1]

    def test_calculate_check_shape(self):
        check = True
        for i in range(len(self.sample_nodes) - 1):
            check &= (len(self.perceptrons.layer(i + 1)), 1) == self.perceptrons.calculate(step=i).shape

        assert check

    def test_calculate_all(self):
        check = True

        self.perceptrons.calculate_all()

        for layer in self.perceptrons.layers[2:]:
            row, _ = layer.shape
            for i in range(row):
                check &= (layer.item(i, 0) != 0.0)

        assert check

    def test_err_shape(self):
        self.perceptrons.calculate_all()

        check = True
        for i in range(len(self.perceptrons.weights)):
            for output_node in self.perceptrons.layer(i):
                assert isinstance(self.perceptrons.err(i, 0)(output_node.item(0, 0)), float)

        assert check

    def test_delta_shape(self):
        check = True
        for step in range(len(self.perceptrons.weights)):
            check &= (self.perceptrons.weight(step).shape == self.perceptrons.delta(step, 0).shape)

        assert check

