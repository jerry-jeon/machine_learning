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
        fake_data = np.array([[5.332, 1.233, 8.66]]).T
        fake_weight = self.perceptrons.beginning_weight(3, 1)
        assert 0 < sigmoid(fake_weight, fake_data) < 1

    def test_initialize_layer_number(self):
        assert len(self.perceptrons.layers) == len(self.sample_nodes)

    def test_initialize_layer_shape(self):
        for index, node_length in enumerate(self.sample_nodes):
            assert ((node_length, 1) == self.perceptrons.layers[index].shape)

    def test_augmented_layer_length_same_as_node_length_plus_one(self):
        for index, node_length in enumerate(self.sample_nodes):
            self.perceptrons.change_layer(index, np.array([1.0] * node_length).reshape((node_length, 1)))
            assert ((node_length + 1, 1) == self.perceptrons.augmented_layer(index).shape)

    def test_augmented_layer_first_node_value_is_one(self):
        for index, node_length in enumerate(self.sample_nodes):
            self.perceptrons.change_layer(index, np.array([3.0] * node_length).reshape((node_length, 1)))
            assert (1.0 == self.perceptrons.augmented_layer(index)[0, 0])

    def test_initialize_weight_number(self):
        assert len(self.perceptrons.weights) == len(self.sample_nodes) - 1

    def test_initialize_weight_shape(self):
        for i in range(len(self.sample_nodes) - 1):
            assert ((self.sample_nodes[i] + 1, self.sample_nodes[i+1]) == self.perceptrons.weight(i).shape)

    def test_initialize_weight_range(self):
        for weight in self.perceptrons.weights:
            row, col = weight.shape
            for i in range(row):
                for j in range(col):
                    value = weight.item(i, j)
                    assert (value >= -0.01) & (value <= 0.01) & (value != 0)

    def test_last_layer(self):
        assert self.perceptrons.last_layer() == self.perceptrons.layers[len(self.perceptrons.layers) - 1]

    def test_calculate_check_shape(self):
        for i in range(len(self.sample_nodes) - 1):
            assert (len(self.perceptrons.layers[i + 1]), 1) == self.perceptrons.calculate(step=i).shape

    def test_calculate_all(self):
        self.perceptrons.calculate_all()

        for layer in self.perceptrons.layers[2:]:
            row, _ = layer.shape
            for i in range(row):
                assert (layer.item(i, 0) != 0.0)

    def test_err_shape(self):
        #remove dependency with calculate_all
        self.perceptrons.actual_class = 0
        self.perceptrons.calculate_all()

        for i in range(len(self.perceptrons.weights)):
            for output_node in self.perceptrons.augmented_layer(i):
                #TODO refactor output_node[0]
                assert isinstance(self.perceptrons.err(i)(output_node[0]), float)

    def test_delta_shape(self):
        self.perceptrons.actual_class = 0
        for step in range(len(self.perceptrons.weights)):
            weight_shape = self.perceptrons.weight(step).shape
            delta_shape = self.perceptrons.delta_matrix(step).shape
            assert (weight_shape == delta_shape)

