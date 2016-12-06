from unittest import TestCase
from functools import reduce
import os

from hw import DeepLearningMachine

TEST_FILE = os.path.join(os.path.dirname(__file__), 'test_trn.txt')

class TestDeepLearningMachine(TestCase):

    def setUp(self):
        self.machine = DeepLearningMachine()
        self.fake_data = [1] * 14

    def test_initialize(self):
        assert isinstance(self.machine, DeepLearningMachine)

    def test_predict_positive_discriminant_class1(self):
        self.machine.discriminant = lambda data : 1
        assert self.machine.predict(self.fake_data) == 1
        self.machine.discriminant = lambda data : 0.7
        assert self.machine.predict(self.fake_data) == 1
        self.machine.discriminant = lambda data : 0.55
        assert self.machine.predict(self.fake_data) == 1

    def test_predict_negative_discriminant_class0(self):
        self.machine.discriminant = lambda data : 0
        assert self.machine.predict(self.fake_data) == 0
        self.machine.discriminant = lambda data : 0.3
        assert self.machine.predict(self.fake_data) == 0
        self.machine.discriminant = lambda data : 0.4
        assert self.machine.predict(self.fake_data) == 0

    def test_predict_negative_discriminant_with_bigpositivethreshold_class1(self):
        self.machine.discriminant = lambda data : 0

        assert self.machine.predict(self.fake_data, 100) == 1

    def test_fileToData(self):
        with open(TEST_FILE) as file:
            training_data = self.machine.file_to_data(file)

        cls_positive = sum(x['cls'] == 1 for x in training_data)
        cls_negative = sum(x['cls'] == 0 for x in training_data)
        cls_invalid = sum(x['cls'] != 0 and x['cls'] != 1 for x in training_data)

        assert len(training_data) == 4
        assert cls_positive == 2
        assert cls_negative == 2
        assert cls_invalid == 0

'''
    def test_converge_epoch_nine_false(self):
        for _ in range(8):
            self.machine.converge()

        assert not self.machine.converge()

    def test_converge_epoch_ten_true(self):
        for _ in range(9):
            self.machine.converge()

        assert self.machine.converge()
'''
