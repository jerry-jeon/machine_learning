from math import log, sqrt
vars = [[], []]
with open('data/trn.txt') as file:
    data = file.readlines()

    for line in data:
        datas = line.split()
        if len(datas) > 13:
            result = int(datas.pop())
            vars[result].append(datas)

all = len(vars[0]) + len(vars[1])
priors = []
for classes in vars:
    priors.append(len(classes) / (all * 1.0))
    
sumy =  [[], []]
for i in range(0, 13):
    sumy[0].append(0.0)
    sumy[1].append(0.0)
means = [[], []]
for result, datas in enumerate(vars):
    for line, event in enumerate(datas):
        for index, datum in enumerate(event):
            sumy[result][index] += float(datum)

        

for result, datas in enumerate(sumy):
    size = len(vars[result]) * 1.0
    for sum in datas:
        means[result].append(sum / size)

variances = [[0.0] * 13, [0.0] * 13]
for result, datas in enumerate(vars):
    size = len(vars[result]) * 1.0
    for line, event in enumerate(datas):
        for index, datum in enumerate(event):
            variances[result][index] += pow(float(datum) - means[result][index], 2) / size

def g(x, result, index):
    return -log(sqrt(variances[result][index])) - (pow((x - means[result][index]), 2)) / (2 * variances[result][index])

def clssifier(x):
    total = [0] * 2
    for result in range(2):
        for index in range(13):
            total[result] += g(x[index], result, index)
        total[result] += log(priors[result])
   
    if total[0] > total[1]:
        return 0
    else:
        return 1

test_datas = []
with open('data/tst.txt') as file:
    data = file.readlines()

    for line in data:
        test_datas.append(line.split())

correct = 0
tp = 0
tn = 0
fp = 0
fn = 0

for data in test_datas:
    if len(data) >= 13: 
        x = list(map(lambda x : float(x), data[:13]))
        predict = clssifier(x);
        actual = int(data[13])
        if predict == actual:
            correct += 1
            if predict == 0:
                tp += 1
            else:
                tn += 1
        else:
            if predict == 0:
                fn += 1
            else:
                fp += 1



cp = tp + fn * 1.0
cn = tn + fp * 1.0
print(correct)
print("TP : ", tp)
print("FP : ", fp)
print("TPR : ", (tp / cp))
print("FPR : ", (fp / cn))
print(len(test_datas))
