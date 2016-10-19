from math import log, sqrt
from operator import add
import numpy as np

# consider assume class size is two.
ATTRIBUTE_SIZE = 13
CLS_SIZE = 2
class Machine:
    def __init__(self):
        self.training_data = []
        self.cov_mat = []
        self.cls_size = [0] * CLS_SIZE

    def is_valid(self, data):
        if len(data) > 13:
            return True
        else:
            return False

# hmm.. I think parameter should be detail
    def learnFile(self, file):
        data_lines = file.readlines()
        sum = np.full((CLS_SIZE, ATTRIBUTE_SIZE), 0.0)
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                # TODO rename ho
                data = {
                    'cls': int(data_line.pop()),
                    'data': np.array([float(i) for i in data_line]),
                }
                sum[data['cls']] = np.add(sum[data['cls']], data['data'])
                self.cls_size[data['cls']] += 1
                self.training_data.append(data)

        self.calculateMean(sum)
        self.calculateCovarianceMatrix()
        
    def calculateParameters(self):
        self.calculateVariance()

    def calculateMean(self, sum):
        for cls, cls_sum in enumerate(sum):
            cls_sum /= self.cls_size[cls]

        self.mean = np.mat(sum).T

    def calculateCovarianceMatrix(self):
        sum = [np.full((ATTRIBUTE_SIZE, ATTRIBUTE_SIZE), 0.0)] * CLS_SIZE
        for data in self.training_data:
            sum[data['cls']] += np.mat(data['data']).T * np.mat(data['data'])

        for i in range(CLS_SIZE):
            sum[i] = sum[i] / self.cls_size[i]


        for cls in range(CLS_SIZE):
            mean = self.mean[:,cls] * self.mean[:,cls].T
            self.cov_mat.append(sum[cls] - mean)

    def g(self, x, cls):
        w_1 = -0.5 * self.cov_mat[cls].I
        w_2 = self.cov_mat[cls].I * self.mean[:,cls]
        w_3 = -0.5 * self.mean[:,cls].T * self.cov_mat[cls].I * self.mean[:,cls] - 0.5*log(np.linalg.det(self.cov_mat[cls])) + log(self.cls_size[cls] / (len(self.training_data) * 1.0))
        result = np.mat(x) * w_1 * np.mat(x).T + w_2.T * np.mat(x).T + w_3

        return result[0]

    def predict(self, data):
        results = []
        for i in range(CLS_SIZE):
            results.append(self.g(data, i))

        return results.index(max(results))


    def predictFile(self, file):
        true_positive = true_negative = false_positive = false_negative = size = 0
        data_lines = file.readlines()
        for event in data_lines:
            data_line = event.split()
            if self.is_valid(data_line):
                size += 1
                actual_cls = int(data_line.pop())
                predict = self.predict(np.array([float(i) for i in data_line]))
                if predict == actual_cls:
                    if predict == 0:
                        true_positive += 1
                    else:
                        true_negative += 1
                else:
                    if predict == 0:
                        false_negative += 1
                    else:
                        false_positive += 1
                
        print("Empirical error : " + str(false_negative + false_positive))
        print()
        print("Confusion Matrix")
        print("--------------------------------------------")
        print("True positive : " + str(true_positive))
        print("True negative : " + str(true_negative))
        print("False positive : " + str(false_positive))
        print("False negative : " + str(false_negative))


machine = Machine()
with open('data/trn.txt') as file:
    machine.learnFile(file)

test_datas = []
with open('data/tst.txt') as file:
    machine.predictFile(file)
