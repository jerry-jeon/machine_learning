from unittest import TestCase

from hw import Perceptrons
import numpy as np


class TestPercentrons(TestCase):

    def setUp(self):
        self.sample_nodes = [13, 2, 2, 1]
        self.perceptrons = Perceptrons(self.sample_nodes)

    def test_initialize_layer_number(self):
        assert len(self.perceptrons.layers) == len(self.sample_nodes)

    def test_initialize_layer_shape(self):
        check = True
        for index, node_length in enumerate(self.sample_nodes):
            check &= ((node_length, 1) == self.perceptrons.layer(index).shape)

        assert check

    def test_initialize_weight_number(self):
        assert len(self.perceptrons.weights) == len(self.sample_nodes) - 1

    def test_initialize_weight_shape(self):
        check = True

        for i in range(len(self.sample_nodes) - 1):
            check &= ((self.sample_nodes[i], self.sample_nodes[i+1])
                       == self.perceptrons.weight(i).shape)

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

    def test_last_layer(self):
        assert self.perceptrons.last_layer() == self.perceptrons.layers[len(self.perceptrons.layers) - 1]

    def test_calculate_check_shape(self):
        check = True
        for i in range(len(self.sample_nodes) - 1):
            check &= (len(self.perceptrons.layer(i + 1)), 1) == self.perceptrons.calculate(step=i).shape

        assert check

    def test_calculate_all(self):
        check = True

        self.perceptrons.calculateAll()

        for layer in self.perceptrons.layers[2:]:
            row, _ = layer.shape
            for i in range(row):
                check &= (layer.item(i, 0) != 0.0)

        assert check

    def test_generate_delta_shape(self):
        check = True
        for step in range(len(self.perceptrons.weights)):
            check &= (self.perceptrons.weight(step).shape == self.perceptrons.generate_delta(step, 0).shape)

        assert check

    def test_err_shape(self):
        self.perceptrons.calculateAll()

        check = True
        for i in range(len(self.perceptrons.weights)):
            for output_node in self.perceptrons.layer(i):
                assert isinstance(self.perceptrons.err(i, 0)(output_node.item(0, 0)), float)

        assert check



