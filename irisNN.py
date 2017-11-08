import matplotlib.pyplot as plt   # for making plots
import numpy as np  # for everything mathy
import urllib2      # to read the data file from the web
from sklearn.neural_network import MLPClassifier

# Read the text file from the internet into to lists: 
#   data : list of 4-tuples : (sl,sw,pl,pw)
#   lables : the iris species corresponding to each data tuple
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data'
data = []
labels = []
for line in urllib2.urlopen(url):
    line = line.strip() # remove training newline character
    if line == '': # last line of the file is empty
        continue
    tokens = line.split(',') # comma-separated values
    data.append((
        float(tokens[0]), float(tokens[1]), 
        float(tokens[2]), float(tokens[3])
    ))
    labels.append(tokens[4].strip())

# Preprocess the data
data_new = []
for d in data:
    data_new.append(
        #np.array(d) / np.sum(d)
        np.array(d) / np.sqrt(np.sum(np.square(d)))
    )
data = np.array(data_new)

labels = np.array(labels)
#shuffle data so that irises of all types are in training set credit: https://stackoverflow.com/questions/43229034/randomly-shuffle-data-and-labels-from-different-files-in-the-same-order
np.random.seed(3)

idx = np.random.permutation(len(data))
data,labels = data[idx], labels[idx]

np.random.seed()

num_training_samples = 100

#np.random.shuffle(data)

#training data
trData = np.array(data[:100])
trLabels = labels[:100]


# Train the neural network classifier.
clf = MLPClassifier(
    solver='lbfgs',            # lbfgs is good for small problems.
    activation='relu',     # options: logistic, tanh, relu, identity
    hidden_layer_sizes=(4),  # play with this!
    tol=1e-8                  # default: 1e-4
)
clf.fit(trData, trLabels)
print("Training set score: %f (out of 1.0)" % clf.score(trData, trLabels))

#testing data
tData = data[100:]
tLabels = labels[100:]

#clf.fit(tData, tLabels)
print("Testing set score: %f (out of 1.0)" % clf.score(tData, tLabels))

for weight in clf.coefs_:
    print weight
    print "\n"

for bias in clf.coefs_:
    print bias
    print "\n"
