from unittest import TestCase

from hw import Machine


class TestMachine(TestCase):
    def setUp(self):
        self.machine = Machine()
        self.fake_data = [1] * 14

    def test_is_valid_with_invalid_data(self):
        data = [0] * 13
        assert self.machine.is_valid(data) == False

    def test_is_valid_with_valid_data(self):
        #TODO change 13 to ATTIBUTE_SIZE constant
        data = [0] * (13 + 1)
        assert self.machine.is_valid(data) == True

    def test_predict_when_discriminant_return_large_class1(self):
        self.machine.discriminant = lambda x, cls: cls

        assert self.machine.predict(self.fake_data, 0) == 1

    def test_predict_when_discriminant_return_large_class0(self):
        self.machine.discriminant = lambda x, cls: -cls

        assert self.machine.predict(self.fake_data, 0) == 0

    def test_predict_when_discriminant_return_large_class0_with_big_positive_threshold(self):
        self.machine.discriminant = lambda x, cls: -cls

        assert self.machine.predict(self.fake_data, threshold=100) == 1
