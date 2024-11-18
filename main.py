import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(15)

data = pd.read_csv('data-class.csv', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.hstack((np.ones((X.shape[0], 1)), X))

num_classes = len(np.unique(y))
num_examples, num_predictors = X.shape

train_size = int(0.6 * num_examples)
val_size = int(0.2 * num_examples)
indices = np.random.permutation(num_examples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

theta = np.random.rand(num_classes, num_predictors)
theta[-1, :] = 0

def accuracy(X, y, theta):
    scores = X @ theta.T
    predictions = np.argmax(scores, axis=1)
    accuracy_value = np.mean(predictions == y)
    return accuracy_value


# softmax sa optimizacijom
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# racunanje log-verodostojnosti
def log_likelihood(X, y, theta):
    m = X.shape[0]  # number of examples
    scores = X @ theta.T
    correct_class_scores = scores[np.arange(m), y]  # select correct class using y
    log_sum_exp = np.log(np.sum(np.exp(scores), axis=1))
    log_likelihood_value = np.sum(correct_class_scores - log_sum_exp)
    return log_likelihood_value


# racunanje gradijenta
def gradient(X, y, theta):
    scores = X @ theta.T
    m = X.shape[0]
    grad = np.zeros_like(theta)
    for l in range(num_classes):
        ind = [int(j == l) for j in y]
        for i in range(m):
            grad[l] += (ind[i] - np.exp(scores[i][l]) / np.sum(np.exp(scores[i]))) * X[i]
    return grad

def gradient_descent(_X, _y, _X_val, _y_val, _theta, _alpha, _batch_size, patience=10):
    m = _X.shape[0]
    log_likelihood_history = []
    samples_seen = []
    total_samples = 0
    best_theta = np.copy(theta)
    # Inicijalno računanje log-verodostojnosti na validacionom skupu
    best_ll = log_likelihood(_X_val, _y_val, _theta)
    log_likelihood_history.append(best_ll)
    samples_seen.append(0)

    no_improvement_count = 0  # Broj uzastopnih iteracija bez poboljšanja

    while no_improvement_count < patience:
        indices = np.random.permutation(m)
        _X_shuffled = _X[indices]
        y_shuffled = _y[indices]

        for j in range(0, m, _batch_size):
            end = min(j + _batch_size, m)
            _X_batch = _X_shuffled[j:end]
            y_batch = y_shuffled[j:end]

            # Ažuriranje theta koristeći gradijent
            _theta += _alpha * gradient(_X_batch, y_batch, _theta)
            _theta[-1] = 0  # Zadnji red postavljamo na nule

            # Računanje log-verodostojnosti na validacionom skupu
            ll = log_likelihood(_X_val, _y_val, _theta)
            log_likelihood_history.append(ll)

            total_samples += (end - j)
            samples_seen.append(total_samples)

            # Provera da li je log-verodostojnost bolja
            if ll > best_ll:
                best_ll = ll
                best_theta = _theta
                no_improvement_count = 0  # Resetujemo brojač jer smo našli poboljšanje
            else:
                no_improvement_count += 1  # Ako nema poboljšanja, povećavamo brojač

            # Ako nema poboljšanja nakon definisanog broja iteracija, zaustavljamo algoritam
            if no_improvement_count >= patience:
                return best_theta, log_likelihood_history, samples_seen

    return best_theta, log_likelihood_history, samples_seen




# pocetno stanje
print(f"Pocetna log verodostojnost: {log_likelihood(X_test, y_test, theta)}")
print(f"Pocetna tacnost: {accuracy(X_test, y_test, theta)}")

# optimalni parametri
batch_size = 16
alpha = 0.3384857145670226

# treniranje
theta, log_history, samples_seen = gradient_descent(
    X_train, y_train, X_val, y_val, theta, _alpha=alpha, _batch_size=batch_size, patience=5
)

best_log = log_history[np.argmax(log_history)]
best_sample = samples_seen[np.argmax(log_history)]
plt.figure(figsize=(10, 6))
plt.plot(samples_seen, log_history, label='Log-verodostojnost')
plt.scatter(best_sample, best_log, color='red', zorder=5, label=f'Najbolja log-verodostojnost: {best_log:.2f} pri {best_sample} uzoraka')
plt.xlabel('Broj obrađenih uzoraka')
plt.ylabel('Log-verodostojnost')
plt.title('Log-verodostojnost u zavisnosti od broja uzorka')
plt.legend()
plt.show()

# pisanje rezultata
print(f'Log verodostojnost na treningu: {log_likelihood(X_train, y_train, theta)}')
print(f'Tacnost na trening: {accuracy(X_train, y_train, theta)}')
print(f'Log verodostojnost na testu: {log_likelihood(X_test, y_test, theta)}')
print(f'Tacnost na testu: {accuracy(X_test, y_test, theta)}')