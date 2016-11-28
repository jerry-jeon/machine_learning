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

class DeepLearningMachine(Machine):

    def __init__(self):
        self.epoch = 0
    
    def beginningWeight(self, row, col):
        return np.random.uniform(-0.01, 0.01, (row, col)) 

    def converge(self, delta = 0):
        self.epoch += 1
        if self.epoch > 10:
            return True
        else:
            return False
        '''
                    if predelta == (delta  > 0):
                        change += 1
                    else:
                        change = 0
                        
                    predelta = delta > 0
                    '''

    def learnFile(self, file):
        training_data = self.fileToData(file)

        hid_node = 2
        w = self.beginningWeight(hid_node, ATTRIBUTE_SIZE)
        v = self.beginningWeight(1, hid_node)
        
        eta = 0.001 # learning rate
        activation = self.linear;
        h = np.full((1, hid_node), 0.0)
        z = np.full((1, hid_node), 0.0)

        while not self.converge():
            for data in training_data:
                for h in range(hid_node):
                    #print(-(np.mat(w[h]) * np.mat(data['data']).T))
                    try:
                        z[:,h] = 1.0 / (1 + exp(-(np.mat(w[h]) * np.mat(data['data']).T)))
                    except:
                        print(-(np.mat(w[h]) * np.mat(data['data']).T))

                y = np.mat(v) * np.mat(z).T

                v_delta = eta * (data['cls'] - y) * np.mat(z)

                for h in range(hid_node):
                    w_delta = eta * ((data['cls'] - y) * v[:,h]) * z[:,h] * (1 - z[:,h]) * np.mat(data['data'].T)

                v += v_delta
                for h in range(hid_node):
                    w[:h,] += w_delta

        def g(x):
            return np.mat(v) * np.mat(w) * np.mat(x).T

        self.discriminant = g

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
        cls_size = [0] * CLS_SIZE
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


machine = DeepLearningMachine()
with open('data/trn.txt') as file:
    machine.learnFile(file)

test_datas = []
with open('data/tst.txt') as file:
    machine.predictFile(file, False)
