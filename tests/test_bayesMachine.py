from unittest import TestCase

from hw import BayesMachine

class TestBayesMachine(TestCase):
    def setUp(self):
        self.machine = BayesMachine()
        self.fake_data = [1] * 14

    def test_predict_when_discriminant_return_large_class1(self):
        self.machine.discriminant = lambda x, cls: cls

        assert self.machine.predict(self.fake_data, 0) == 1

    def test_predict_when_discriminant_return_large_class0(self):
        self.machine.discriminant = lambda x, cls: -cls

        assert self.machine.predict(self.fake_data, 0) == 0

    def test_predict_when_discriminant_return_large_class0_with_big_positive_threshold(self):
        self.machine.discriminant = lambda x, cls: -cls

        assert self.machine.predict(self.fake_data, threshold=100) == 1
