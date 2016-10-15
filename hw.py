from math import log
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
    
print priors
# TODO buckets = [0] * 100
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
    for line, event in enumerate(datas):
        for index, datum in enumerate(event):
            variances[result][index] += pow(float(datum) - means[result][index], 2)

for result, datas in enumerate(variances):
    size = len(vars[result]) * 1.0
    map(lambda x: x / size, datas)

print means
print variances

def g(x):
    return -log(sqrt(v)) - (pow((x-m), 2)) / (2 * v) + log(prior)
