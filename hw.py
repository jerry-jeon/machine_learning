from math import log, sqrt
from operator import add
import numpy as np

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

    def predictFile(self, file):
        data_lines = file.readlines()
        original = self.predictDataLines(data_lines)
        original.print()
        original.printRocPoint()

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
            print()

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

    def predict(self, data, threshold):
        positive = self.discriminant(data, 1)
        negative = self.discriminant(data, 0)

        if positive + threshold > negative:
            return 1
        else:
            return 0

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


machine = Machine()
with open('data/trn.txt') as file:
    machine.learnFile(file)

test_datas = []
with open('data/tst.txt') as file:
    machine.predictFile(file)
