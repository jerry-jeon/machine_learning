from math import log, sqrt, exp
from operator import add
import numpy as np
from functools import reduce
import sys

#consider expand abstract attribute size, class size
ATTRIBUTE_SIZE = 13
CLS_SIZE = 2
logging = False


class Machine:

    def is_valid(self, data):
        if len(data) > 13:
            return True
        else:
            return False

    def predict_file(self, data_file, with_roc):
        roc_file = open("roc.txt", "w")
        data_lines = data_file.readlines()
        original = self.predict_data_lines(data_lines)
        original.print()
        original.print_roc_point()

        if with_roc:
            print()
            print("Give threshold for draw roc curve....")
            print()

            for i in range(200):
                threshold = -10 + (i / 10)
                print("threshold : " + str(threshold))
                result = self.predict_data_lines(data_lines, threshold)
                if result.is_eer():
                    EER = result
                result.print_roc_point()
                roc_file.write(str(result.fp_rate()) + "\t" + str(result.tp_rate()) + "\n")
                print()

            roc_file.close()

            try:
                print("Equal error rate")
                EER.print_roc_point()
            except:
                print("Program can't find EER")

    def predict_data_lines(self, data_lines, threshold = 0):
        predictResult = PredictResult()
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                actual_cls = int(data_line.pop())
                predict = self.predict(np.array([float(i) for i in data_line]), threshold)
                predictResult.add_data(predict, actual_cls)

        return predictResult


class BayesMachine(Machine):

    def learn_file(self, file):
        training_data, sum, cls_size, trans = self.file_to_data(file)
        mean = self.calculate_mean(sum, cls_size)
        cov_mat = self.calculate_covariance_matrix(mean, training_data, cls_size, trans)
        prior = [cls_size[cls] / (len(training_data) * 1.0) for cls in range(CLS_SIZE)]

        self.discriminant = self.make_discriminant(cov_mat, mean, prior)

    def file_to_data(self, file):
        training_data = []
        cls_size = [0] * CLS_SIZE
        data_lines = file.readlines()
        sum = np.full((CLS_SIZE, ATTRIBUTE_SIZE), 0.0)
        trans = [np.full((ATTRIBUTE_SIZE, ATTRIBUTE_SIZE), 0.0)] * CLS_SIZE

        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                data = {
                    'cls': int(data_line.pop()),
                    'data': np.array([float(i) for i in data_line]),
                }
                sum[data['cls']] = np.add(sum[data['cls']], data['data'])
                trans = np.add(trans[data['cls']], np.mat(data['data']).T * np.mat(data['data']))
                cls_size[data['cls']] += 1
                training_data.append(data)
        return training_data, sum, cls_size, trans
        
    def calculate_mean(self, sum, cls_size):
        means = []
        for cls, cls_sum in enumerate(sum):
            mean = cls_sum / cls_size[cls]
            means.append(np.mat(mean).T)

        return means

    def calculate_covariance_matrix(self, mean, training_data, cls_size, trans):
        cov_mat = [np.full((ATTRIBUTE_SIZE, ATTRIBUTE_SIZE), 0.0)] * CLS_SIZE
        for data in training_data:
            cov_mat[data['cls']] = np.add(cov_mat[data['cls']], (np.mat(data['data']).T - mean[data['cls']]) * (np.mat(data['data']).T - mean[data['cls']]).T)

        for i in range(CLS_SIZE):
            cov_mat[i] = cov_mat[i] / cls_size[i]
        
        return cov_mat

    def make_discriminant(self, cov_mat, mean, prior):
        def g(x, cls):
            w_1 = -0.5 * cov_mat[cls].I
            w_2 = cov_mat[cls].I * mean[cls]
            w_3 = -0.5 * mean[cls].T * cov_mat[cls].I * mean[cls] - (0.5 * log(np.linalg.det(cov_mat[cls]))) + log(prior[cls])

            result = np.mat(x) * w_1 * np.mat(x).T + w_2.T * np.mat(x).T + w_3

            return result[0]

        return g

    def predict(self, data, threshold):
        positive = self.discriminant(data, 1)
        negative = self.discriminant(data, 0)

        if positive + threshold > negative:
            return 1
        else:
            return 0


def sigmoid(weight, values):
    return 1.0 / (1 + exp(-(weight.T @ values)))


class Perceptrons():

    def __init__(self, nodes):
        self.learning_rate = 0.001
        self.layers = [None] * len(nodes)
        self.weights = []
        self.augmented_layers = [None] * len(nodes)

        for index, node_length in enumerate(nodes):
            self.change_layer(index, np.full((node_length, 1), 0.0))

        for i in range(len(nodes) - 1):
            self.weights.append(self.beginning_weight(nodes[i] + 1, nodes[i + 1]))

    def last_layer(self):
        return self.layers[len(self.layers) - 1]

    def augmented_layer(self, index):
        return self.augmented_layers[index]

    def change_layer(self, index, layer):
        self.layers[index] = layer
        augmented_shape = (self.layers[index].shape[0] + 1, 1)
        self.augmented_layers[index] = np.append([1.0], self.layers[index]).reshape(augmented_shape)

    def weight(self, index):
        return self.weights[index]

    def beginning_weight(self, row, col):
        return np.random.uniform(-0.01, 0.01, (row, col))

    def calculate(self, step):
        results = []
        for cls in range(len(self.layers[step + 1])):
            results.append(sigmoid(self.weight(step)[:, [cls]], self.augmented_layer(step)))
            # TODO refactoring
        return np.array([results]).T

    def calculate_all(self):
        for step in range(len(self.weights)):
            self.change_layer(step + 1,  self.calculate(step))

    def err(self, weight_index):
        if weight_index == len(self.weights) - 1:
            return lambda output_node: self.actual_class - self.last_layer()[0][0]
        else:
            above_err = self.err(weight_index + 1)
            above_layer = self.augmented_layer(weight_index + 1)

            #err_sum = reduce(lambda x, y: x + above_err(i) * y, above_layer, 0)
            err_sum = 0.0
            for i in range(len(above_layer)):
                #TODO should refactoring append [0] behind column vector
                err_sum += (above_err(i) * above_layer[i][0])
#make code looks good
            return lambda output_node: err_sum * (self.augmented_layer(weight_index + 1)[output_node] @ (1 - self.augmented_layer(weight_index + 1)[output_node]))

    def delta(self, step, above_node_index, below_node_index):
        result = self.learning_rate * self.err(step)(above_node_index) * self.augmented_layer(step)[below_node_index]
        if isinstance(result, float):
            return result
        else:
            return result[0]

    def delta_matrix(self, step):
        yop = []
        for below_node_index in range(len(self.augmented_layer(step))):
            results = []
            for above_node_index in range(len(self.layers[step + 1])):
                results.append(self.delta(step, above_node_index, below_node_index))
            yop.append(results)

        return np.array(yop)

    def update_weight(self, step):
        result = self.delta_matrix(step)
        self.weights[step] += result

    def update_weight_all(self):
        for i in range(len(self.weights)):
            self.update_weight(i)

    def back_propogation(self, data):
        self.change_layer(0, data['data'])
        self.actual_class = data['cls']
        self.calculate_all()
        self.update_weight_all()

    def info(self):
        information = ""
        information += "Layer depth : " + str(len(self.layers)) + "\n"
        information += "Weight number : " + str(len(self.weights)) + "\n"
        information += "Layers" + "\n"
        information += "===========================" + "\n"
        information += str(self.layers) + "\n"
        information += "Weights" + "\n"
        information += "===========================" + "\n"
        information += str(self.weights) + "\n"
        return information

class DeepLearningMachine(Machine):

    def __init__(self, perceptrons):
        self.epoch = 0
        self.layers = []
        self.weights = []
        self.training_data = []
        self.perceptrons = perceptrons

    def predict(self, data):
        result = self.discriminant(data)
        print("Result : " + str(result))

        if result > 0.5:
            return 1
        else:
            return 0

    def learn_file(self, file):
        self.read_file(file)
        #print(self.perceptrons.info())
        while self.converge():
            print("EPOCH : " + str(self.epoch))
            for data in self.training_data:
                self.perceptrons.back_propogation(data)
        #print(self.perceptrons.info())

        def g(data):
            self.perceptrons.layers[0] = np.array(data)
            self.perceptrons.calculate_all()
            return self.perceptrons.last_layer().item(0, 0)

        self.discriminant = g

    def add_training_data(self, data, cls):
        data = {
            'cls': cls,
            'data': np.array([float(i) for i in data]).reshape((13, 1)),
        }
        self.training_data.append(data)

    def read_file(self, data_file):
        data_lines = data_file.readlines()

        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                cls = int(data_line.pop())
                self.add_training_data(data_line, cls)

    def converge(self, delta = 0):
        self.epoch += 1
        if self.epoch >= 3:
            return False
        else:
            return True


class PredictResult:
    def __init__(self):
        self.true_positive = self.true_negative = self.false_positive = self.false_negative = 0

    def add_data(self, predict, actual_cls):
        if predict == actual_cls:
            if predict == 1:
                self.true_positive += 1
            else:
                self.true_negative += 1
        else:
            if predict == 1:
                self.false_positive += 1
            else:
                self.false_negative += 1

    def empirical_error(self):
        return self.false_negative + self.false_positive

    def size(self):
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def fp_rate(self):
        if self.false_positive + self.true_negative <= 0:
            return 0
        return self.false_positive / (self.false_positive + self.true_negative * 1.0)

    def tp_rate(self):
        if self.true_positive + self.false_negative <= 0:
            return 0
        return self.true_positive / (self.true_positive + self.false_negative * 1.0)

    def is_eer(self):
        if 0.99 < self.tp_rate() + self.fp_rate() < 1.01:
            return True
        else:
            return False

    def print_roc_point(self):
        print("--------------------------------------------")
        print("FPR : " + str(self.fp_rate()))
        print("TPR : " + str(self.tp_rate()))

    def print(self):
        print()
        print("Result")
        print("--------------------------------------------")
        print("Empirical error : " + str(self.empirical_error()))
        print("Empirical error : " + str(self.empirical_error() / (self.size() * 1.0)))
        print()
        print("Confusion Matrix")
        print("--------------------------------------------")
        print("True positive : " + str(self.true_positive))
        print("True negative : " + str(self.true_negative))
        print("False positive : " + str(self.false_positive))
        print("False negative : " + str(self.false_negative))


if __name__ == "__main__":
    machine = DeepLearningMachine(Perceptrons([13, 2, 2, 1]))
    with open('data/trn.txt') as file:
        machine.learn_file(file)

    test_datas = []
    with open('data/tst.txt') as file:
        machine.predict_file(file, False)
