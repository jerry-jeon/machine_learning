from math import log, sqrt, exp
from operator import add
import numpy as np
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

    def predictFile(self, file, withRoc):
        roc_file = open("roc.txt", "w")
        data_lines = file.readlines()
        original = self.predictDataLines(data_lines)
        original.print()
        original.printRocPoint()

        if withRoc:
            print()
            print("Give threshold for draw roc curve....")
            print()

            for i in range(200):
                threshold = -10 + (i / 10)
                print("threshold : " + str(threshold))
                result = self.predictDataLines(data_lines, threshold)
                if result.is_EER():
                    EER = result
                result.printRocPoint()
                roc_file.write(str(result.fp_rate()) + "\t" + str(result.tp_rate()) + "\n")
                print()

            roc_file.close()

            try:
                print("Equal error rate")
                EER.printRocPoint()
            except:
                print("Program can't find EER")

    def predictDataLines(self, data_lines, threshold = 0):
        predictResult = PredictResult()
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                actual_cls = int(data_line.pop())
                predict = self.predict(np.array([float(i) for i in data_line]), threshold)
                predictResult.addData(predict, actual_cls)

        return predictResult

class BayesMachine(Machine):

    def learnFile(self, file):
        training_data, sum, cls_size, trans = self.fileToData(file)
        mean = self.calculateMean(sum, cls_size)
        cov_mat = self.calculateCovarianceMatrix(mean, training_data, cls_size, trans)
        prior = [cls_size[cls] / (len(training_data) * 1.0) for cls in range(CLS_SIZE)]

        self.discriminant = self.makeDiscriminant(cov_mat, mean, prior)

    def fileToData(self, file):
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
        

    def calculateMean(self, sum, cls_size):
        means = []
        for cls, cls_sum in enumerate(sum):
            mean = cls_sum / cls_size[cls]
            means.append(np.mat(mean).T)

        return means

    def calculateCovarianceMatrix(self, mean, training_data, cls_size, trans):
        cov_mat = [np.full((ATTRIBUTE_SIZE, ATTRIBUTE_SIZE), 0.0)] * CLS_SIZE
        for data in training_data:
            cov_mat[data['cls']] = np.add(cov_mat[data['cls']], (np.mat(data['data']).T - mean[data['cls']]) * (np.mat(data['data']).T - mean[data['cls']]).T)

        for i in range(CLS_SIZE):
            cov_mat[i] = cov_mat[i] / cls_size[i]
        
        return cov_mat

    def makeDiscriminant(self, cov_mat, mean, prior):
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

def sigmoid(self, weight, values):
    return 1.0 / (1 + exp(-(weight.T * values)))


class Layer():

    def __init__(self, node_number):
        self.nodes = np.full((1, node_number), 0).T

    def calculate(self):
        if self.is_last:
            return self.weight.T * self.values
        else:
            return sigmoid(self.weight, self.values)

    def print(self):
        print(-(self.weight.T * self.values))


class Weight():

    def __init__(self, row, col):
        self.mat = self.beginningWeight(row, col)

    def beginningWeight(self, row, col):
        return np.random.uniform(-0.01, 0.01, (row, col))

    def __getitem__(self, index):
        return self.mat[index]


class DeepLearningMachine(Machine):

    def __init__(self):
        self.epoch = 0
        self.layers = []
        self.weights = []

    def predict(self, data, threshold):
        result = self.discriminant(data)

        if result + threshold > 0:
            return 1
        else:
            return 0

    def linear(self, O, cls):
        return O[cls]

    def sigmoid(self, O, cls):
        return 1.0 / (1.0 + exp(O[cls]))

    def softmax(self, O, cls):
        denominator = 0.0
        for cls in range(CLS_SIZE):
            denominator += exp(O[cls])
        return exp(O[cls]) / denominator

    def fileToData(self, file):
        training_data = []
        data_lines = file.readlines()

        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                data = {
                    'cls': int(data_line.pop()),
                    'data': np.array([float(i) for i in data_line]),
                }
                training_data.append(data)

        return training_data

    def converge(self, delta = 0):
        self.epoch += 1
        if self.epoch >= 10:
            return True
        else:
            return False

'''
    def makeLayers(self, nodes): #suppose nodes is int array
        for i in range(len(nodes)):
            node_number = nodes[i]
            layer = Layer(node_number)

            if 'last_layer' in locals():
                last_layer.top_layer = layer

            if i < len(nodes) - 1:
                next_node = nodes[i + 1]
                weight = Weight(next_node_number, node_number)
                weight.input_dimension = layer
                layer.top_weight = weight
                self.weights.append(weight)
            else:
                #TODO 이부분 맘에 안듬
                layer.is_last = True

            if 'last_weight' in locals():
                layer.bottom_weight = last_weight
                last_weight.output_class = layer

            last_weight = weight
            last_layer = layer

            self.layers.append(layer)




    def learnFile(self, file):
        training_data = self.fileToData(file)
        eta = 0.001 # learning rate

        self.makeLayers([13, 2, 2, 1])

        for layer in self.layers:
            print(layer.node)

        while not self.converge():
            for data in training_data:
                layers[0].values = np.mat(data).T
                for layer in layers[:-1]
                    try:
                        layer.top_layer.values = layer.calculate()
                    except:
                        layer.print()


                for index, node in enumerate(layers):
                    for h in range(node):
                        try:
                            print("z shape : " + str(z[index].shape))
                            print("w shape : " + str(np.mat(w[index][h]).shape))
                            print("x shape : " + str(np.mat(data['data']).shape))
                            z[index][:,h] = 1.0 / (1 + exp(-(np.mat(w[index][h]) * lastz.T)))
                            lastz = np.mat(z[index])
                        except:



        layers = [2, 2]
        w = [None] * len(layers)
        z = [None] * len(layers)
        pre_node = ATTRIBUTE_SIZE
        for index, node in enumerate(layers):
            w[index] = self.beginningWeight(node, pre_node)
            h = 0
            print("W Index " + str(index) + " : " + str(np.mat(w[index]).shape))
            z[index] = np.full((1, node), 0.0)
            pre_node = node

        v = self.beginningWeight(1, pre_node)

        while not self.converge():
            for data in training_data:
                lastz = np.mat(data['data'])
                for index, node in enumerate(layers):
                    for h in range(node):
                        try:
                            print("z shape : " + str(z[index].shape))
                            print("w shape : " + str(np.mat(w[index][h]).shape))
                            print("x shape : " + str(np.mat(data['data']).shape))
                            z[index][:,h] = 1.0 / (1 + exp(-(np.mat(w[index][h]) * lastz.T)))
                            lastz = np.mat(z[index])
                        except:
                            print(-(np.mat(w[index][h]) * np.mat(data['data']).T))

                print("3 : " + str(np.mat(z[-1]).shape))
                y = np.mat(v) * np.mat(z[-1]).T

                def err(pre_err, pre_weight, cur_z):
                    sum = 0
                    for i in range(len(pre_err)): # pre_node 개수
                        print("PRE : " + str(pre_err.shape))
                        print("PRE : " + str(pre_weight.shape))
                        print("PRE : " + str(pre_err[i].shape))
                        print("WEI : " + str(pre_weight[i].shape))
                        sum += pre_err[i] * pre_weight[i]

                    print("SUM : " + str(sum.shape))

                    return sum * cur_z * (1 - cur_z)

                def delta(err, vector):
                    return eta * err * vector

                v_err = (data['cls'] - y)
                v_delta = eta * v_err * np.mat(z[-1])
                print("Vdelta : " + str(v_delta.shape))

                v += v_delta

                pre_err = v_err;
                pre_weight = v;
                cur_z = z[1]


                for index, node in reversed(list(enumerate(layers))):
                    erru = err(pre_err, pre_weight, cur_z)
                    delu = delta(erru, vector)

                    for h in range(node):
                        print(w[index].shape)
                        print(deli.shape)
                        w[index][h] += delu

                    pre_err = erru;
                    pre_weight = w[index]
                    cur_z = z[index + 1]


        def g(x):
            return np.mat(v) * np.mat(w) * np.mat(x).T

        self.discriminant = g
        '''


class PredictResult:
    def __init__(self):
        self.true_positive = self.true_negative = self.false_positive = self.false_negative = 0

    def addData(self, predict, actual_cls):
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

    def empiricalError(self):
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

    def is_EER(self):
        if 0.99 < self.tp_rate() + self.fp_rate() < 1.01:
            return True
        else:
            return False

    def printRocPoint(self):
        print("--------------------------------------------")
        print("FPR : " + str(self.fp_rate()))
        print("TPR : " + str(self.tp_rate()))

    def print(self):
        print()
        print("Result")
        print("--------------------------------------------")
        print("Empirical error : " + str(self.empiricalError()))
        print("Empirical error : " + str(self.empiricalError() / (self.size() * 1.0)))
        print()
        print("Confusion Matrix")
        print("--------------------------------------------")
        print("True positive : " + str(self.true_positive))
        print("True negative : " + str(self.true_negative))
        print("False positive : " + str(self.false_positive))
        print("False negative : " + str(self.false_negative))


if __name__ == "__main__":
    machine = DeepLearningMachine()
    with open('data/trn.txt') as file:
        machine.learnFile(file)

    test_datas = []
    with open('data/tst.txt') as file:
        machine.predictFile(file, False)
