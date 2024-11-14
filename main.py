import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

np.random.seed(7)

data = pd.read_csv('data-class.csv', header=None)

X = data.iloc[:, :-1] # X is everything except last column
y = data.iloc[:, -1] # y is only the last column

X = X.to_numpy()
y = y.to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X) # data needs to be scaled

X = np.hstack((np.ones((X.shape[0], 1)), X)) # adding one column to the left with ones
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
num_predictors = X.shape[1]

theta = np.array([
    [0.52674256, 0.57772689, 0.13140846, 0.32830466, 0.83599875, 0.08346497],
    [0.83909139, 0.43855874, 0.08125015, 0.9565754,  0.3505978,  0.5513],
    [0., 0., 0., 0., 0., 0.]]
)

def accuracy(X, y, theta):
    scores = X @ theta.T
    predictions = np.argmax(scores, axis=1)
    accuracy_value = np.mean(predictions == y)
    return accuracy_value

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def log_likelihood(X, y, theta):
    m = X.shape[0] # number of examples
    scores = X @ theta.T
    correct_class_scores = scores[np.arange(m), y] # select correct class using y
    log_sum_exp = np.log(np.sum(np.exp(scores), axis=1))
    log_likelihood_value = np.sum(correct_class_scores - log_sum_exp)
    return log_likelihood_value

def gradient(X, y, theta):
    scores = X @ theta.T
    m = X.shape[0]
    grad = np.zeros_like(theta)
    for l in range(num_classes):
        ind = [int(j==l) for j in y]
        for i in range(m):
            grad[l]+= (ind[i] - np.exp(scores[i][l])/np.sum(np.exp(scores[i]))) * X[i]
    return  grad

def gradient_descent(_X, _y, _theta, _alpha, _num_iterations, _batch_size):
    m = _X.shape[0]
    log_likelihood_history = []
    samples_seen = []
    total_samples = 0
    ll = log_likelihood(_X, _y, _theta)
    log_likelihood_history.append(ll)
    samples_seen.append(0)

    for i in range(_num_iterations):
        indices = np.random.permutation(m)
        _X_shuffled = _X[indices]
        y_shuffled = _y[indices]

        for j in range(0, m, _batch_size):
            end = min(j + _batch_size, m)
            _X_batch = _X_shuffled[j:end]
            y_batch = y_shuffled[j:end]
            _theta += _alpha * gradient(_X_batch, y_batch, _theta)
            _theta[-1] = 0
            ll = log_likelihood(_X_batch, y_batch, _theta)
            log_likelihood_history.append(ll)
            total_samples += (end - j)
            samples_seen.append(total_samples)

    return _theta, log_likelihood_history, samples_seen


print(log_likelihood(X_train, y_train, theta))
print(accuracy(X_train, y_train, theta))

alpha = 0.1
num_iterations = 1
batch_size = 10

theta, log_likelihood_history, samples_seen = gradient_descent(X_train, y_train, theta, alpha, num_iterations,
                                                               batch_size)

plt.figure(figsize=(10, 6))
plt.plot(samples_seen, log_likelihood_history, label='Log-verodostojnost')
plt.xlabel('Broj obrađenih uzoraka')
plt.ylabel('Log-verodostojnost')
plt.title('Konvergencija modela tokom treninga')
plt.legend()
plt.show()

print(f'Log verodostojnost na treningu: {log_likelihood(X_train, y_train, theta)}')
print(f'Tacnost na trening: {accuracy(X_train, y_train, theta)}')
print(f'Log verodostojnost na testu: {log_likelihood(X_test, y_test, theta)}')
print(f'Tacnost na testu: {accuracy(X_test, y_test, theta)}')
