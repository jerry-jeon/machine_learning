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

