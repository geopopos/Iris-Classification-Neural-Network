import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# A funny function to ground-truth classify training examples
# This is what we're trying to fit with a NN.
def gtLabel(pt):
    pt = np.array(pt) # make sure the point gets treated as an array
    assert len(pt) == 2
    loci = (
        np.array([-0.8, -0.2]),
        np.array([-0.1,  0.5]),
        np.array([0.5,  -0.4])
    ) # list of three 2D points in [-1, 1] x [-1, 1]
    
    # find sum of dist to each locus
    d = []
    for l in loci:
        d.append(np.linalg.norm(pt - l))
        
    # threshold on the sum distance to choose gt category
    if np.average(d) > 0.7 * np.max(d):
        return 1
    else: 
        return 0

# A function to randomly sample the population [-1, 1] x [-1, 1]
def oneSample():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    return np.array([x, y])

# make training data
num_training_samples = 100
np.random.seed(3) # So that we always make the same data points
X_train = [oneSample() for _ in range(num_training_samples)]
Y_train = [gtLabel(x) for x in X_train]
np.random.seed() # re-randomize the rng for later

# Train the neural network classifier.
clf = MLPClassifier(
    solver='lbfgs',            # lbfgs is good for small problems.
    activation='logistic',     # options: logistic, tanh, relu, identity
    hidden_layer_sizes=(8,3),  # play with this!
    tol=1e-10                  # default: 1e-4
)
clf.fit(X_train, Y_train)
print("Training set score: %f (out of 1.0)" % clf.score(X_train, Y_train))

# Plot the result of the classifier
# make these smaller to increase the resolution
n_samples = 80
dx = 2 / float(n_samples)
dy = 2 / float(n_samples)

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-1, 1 + dy, dy),
                slice(-1, 1 + dx, dx)]

# evaluate the classifier on those grid points
z = np.zeros(y.shape)
for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        z[i,j] = clf.predict_proba([[x[i,j], y[i,j]]])[0,0]

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]

fig = plt.figure(figsize=(8,8))
plt.pcolormesh(x, y, z, vmin=0, vmax=1, cmap='coolwarm')
plotsymb = ['or', 'ob'] # lookup list
for x,y in zip(X_train, Y_train):
    plt.plot(x[0], x[1], plotsymb[y], markersize=10, markeredgewidth=2)
