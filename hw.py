from math import log, sqrt
from operator import add
import numpy as np

# consider assume class size is two.
ATTRIBUTE_SIZE = 13
CLS_SIZE = 2
class PredictResult:
    def __init__(self):
        self.true_positive = self.true_negative = self.false_positive = self.false_negative = 0

    def addData(self, predict, actual_cls):
        if predict == actual_cls:
            if predict == 0:
                self.true_positive += 1
            else:
                self.true_negative += 1
        else:
            if predict == 0:
                self.false_negative += 1
            else:
                self.false_positive += 1

    def empiricalError(self):
        return self.false_negative + self.false_positive

    def size(self):
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def fp_rate(self):
        return self.false_positive / (self.false_positive + self.true_negative * 1.0)

    def tp_rate(self):
        return self.true_positive / (self.true_positive + self.false_negative * 1.0)

    def print(self):
        print("Empirical error : " + str(self.empiricalError()))
        print()
        print("Confusion Matrix")
        print("--------------------------------------------")
        print("True positive : " + str(self.true_positive))
        print("True negative : " + str(self.true_negative))
        print("False positive : " + str(self.false_positive))
        print("False negative : " + str(self.false_negative))

#roc curve - x cord is fp-rate y corod - tp rate

class Machine:
    def is_valid(self, data):
        if len(data) > 13:
            return True
        else:
            return False

# hmm.. I think parameter should be detail
    def learnFile(self, file):
        training_data, sum, cls_size = self.fileToData(file)
        mean = self.calculateMean(sum, cls_size)
        cov_mat = self.calculateCovarianceMatrix(mean, training_data, cls_size)
        prior = [cls_size[cls] / (len(training_data) * 1.0) for cls in range(CLS_SIZE)]

        self.discriminant = self.makeDiscriminant(cov_mat, mean, prior)

        self.roc_discriminants = []
        roc_priors = [0.00001, 0.02, 0.3, 0.8, 0.99];
        for roc_prior in roc_priors:
            self.roc_discriminants.append(self.makeDiscriminant(cov_mat, mean, roc_prior))


    def fileToData(self, file):
        training_data = []
        cls_size = [0] * CLS_SIZE
        data_lines = file.readlines()
        sum = np.full((CLS_SIZE, ATTRIBUTE_SIZE), 0.0)
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                data = {
                    'cls': int(data_line.pop()),
                    'data': np.array([float(i) for i in data_line]),
                }
                sum[data['cls']] = np.add(sum[data['cls']], data['data'])
                cls_size[data['cls']] += 1
                training_data.append(data)
        return training_data, sum, cls_size
        

    def calculateMean(self, sum, cls_size):
        for cls, cls_sum in enumerate(sum):
            cls_sum /= cls_size[cls]

        return np.mat(sum).T

    def calculateCovarianceMatrix(self, mean, training_data, cls_size):
        cov_mat = []
        sum = [np.full((ATTRIBUTE_SIZE, ATTRIBUTE_SIZE), 0.0)] * CLS_SIZE
        for data in training_data:
            sum[data['cls']] += np.mat(data['data']).T * np.mat(data['data'])

        for i in range(CLS_SIZE):
            sum[i] = sum[i] / cls_size[i]


        for cls in range(CLS_SIZE):
#TODO
            yo = mean[:,cls] * mean[:,cls].T
            cov_mat.append(sum[cls] - yo)
        
        return cov_mat

    def makeDiscriminant(self, cov_mat, mean, prior):
        def g(x, cls):
            w_1 = -0.5 * cov_mat[cls].I
            w_2 = cov_mat[cls].I * mean[:,cls]
            w_3 = -0.5 * mean[:,cls].T * cov_mat[cls].I * mean[:,cls] - 0.5*log(np.linalg.det(cov_mat[cls])) + log(prior[cls])
            result = np.mat(x) * w_1 * np.mat(x).T + w_2.T * np.mat(x).T + w_3

            return result[0]

        return g

    def predict(self, data, discriminant):
        results = []
        for i in range(CLS_SIZE):
            results.append(discriminant(data, i))

        return results.index(max(results))
    
    def predictDataLines(self, data_lines, discriminant):
        predictResult = PredictResult()
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                actual_cls = int(data_line.pop())
                predict = self.predict(np.array([float(i) for i in data_line]), discriminant)
                predictResult.addData(predict, actual_cls)

        return predictResult


    def predictFile(self, file):
        data_lines = file.readlines()

        return self.predictDataLines(data_lines, self.discriminant)



machine = Machine()
with open('data/trn.txt') as file:
    machine.learnFile(file)

test_datas = []
with open('data/tst.txt') as file:
    predictResult = machine.predictFile(file)
    predictResult.print()
