import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data(filename='EMGsDataset.csv'):
    data = np.genfromtxt(filename, delimiter=',')
    X = data[:2, :].T  # N x p (50000 x 2)
    labels = data[2, :].astype(int)  # N x 1
    Y = np.zeros((len(labels), 5))
    for i, label in enumerate(labels):
        Y[i, label-1] = 1
    return X, Y, labels

def plot_scatter(X, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('Corrugador do Supercílio (Sensor 1)')
    plt.ylabel('Zigomático Maior (Sensor 2)')
    plt.title('Distribuição das Expressões Faciais')
    plt.show()

class MQOClassifier:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, Y):
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ Y
    
    def predict(self, X):
        Y_pred = X @ self.theta
        return np.argmax(Y_pred, axis=1) + 1

class GaussianClassifier:
    def __init__(self):
        self.means = []
        self.covs = []
        self.priors = []
    
    def fit(self, X, Y, labels):
        n_classes = 5
        for c in range(1, n_classes + 1):
            X_c = X[labels == c]
            self.means.append(np.mean(X_c, axis=0))
            self.covs.append(np.cov(X_c.T))
            self.priors.append(len(X_c) / len(X))
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = 5
        scores = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            diff = X - self.means[c]
            cov_inv = np.linalg.pinv(self.covs[c])
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            scores[:, c] = exponent + np.log(self.priors[c])
        
        return np.argmax(scores, axis=1) + 1

class GaussianClassifierEqualCov:
    def __init__(self):
        self.means = []
        self.cov = None
        self.priors = []
    
    def fit(self, X, Y, labels):
        n_classes = 5
        self.cov = np.zeros((X.shape[1], X.shape[1]))
        for c in range(1, n_classes + 1):
            X_c = X[labels == c]
            self.means.append(np.mean(X_c, axis=0))
            self.cov += np.cov(X_c.T) * len(X_c)
            self.priors.append(len(X_c) / len(X))
        self.cov /= len(X)
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = 5
        scores = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            diff = X - self.means[c]
            cov_inv = np.linalg.pinv(self.cov)
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            scores[:, c] = exponent + np.log(self.priors[c])
        
        return np.argmax(scores, axis=1) + 1

class GaussianClassifierAggregated:
    def __init__(self):
        self.means = []
        self.cov = None
        self.priors = []
    
    def fit(self, X, Y, labels):
        n_classes = 5
        self.cov = np.zeros((X.shape[1], X.shape[1]))
        for c in range(1, n_classes + 1):
            X_c = X[labels == c]
            self.means.append(np.mean(X_c, axis=0))
            self.cov += np.cov(X_c.T) * len(X_c)
            self.priors.append(len(X_c) / len(X))
        self.cov /= len(X)
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = 5
        scores = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            diff = X - self.means[c]
            cov_inv = np.linalg.pinv(self.cov)
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            scores[:, c] = exponent + np.log(self.priors[c])
        
        return np.argmax(scores, axis=1) + 1

class NaiveBayesClassifier:
    def __init__(self):
        self.means = []
        self.vars = []
        self.priors = []
    
    def fit(self, X, Y, labels):
        n_classes = 5
        for c in range(1, n_classes + 1):
            X_c = X[labels == c]
            self.means.append(np.mean(X_c, axis=0))
            self.vars.append(np.var(X_c, axis=0) + 1e-10)  # Evita divisão por zero
            self.priors.append(len(X_c) / len(X))
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = 5
        scores = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            diff = X - self.means[c]
            exponent = -0.5 * np.sum((diff ** 2) / self.vars[c], axis=1)
            scores[:, c] = exponent + np.log(self.priors[c])
        
        return np.argmax(scores, axis=1) + 1

class GaussianClassifierRegularized:
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.means = []
        self.covs = []
        self.priors = []
    
    def fit(self, X, Y, labels):
        n_classes = 5
        for c in range(1, n_classes + 1):
            X_c = X[labels == c]
            self.means.append(np.mean(X_c, axis=0))
            cov = np.cov(X_c.T)
            # Regularização de Friedman
            self.covs.append((1 - self.lambda_) * cov + self.lambda_ * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0])
            self.priors.append(len(X_c) / len(X))
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = 5
        scores = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            diff = X - self.means[c]
            cov = self.covs[c] + 1e-4 * np.eye(self.covs[c].shape[0])  # Regularização adicional
            cov_inv = np.linalg.pinv(cov)  # Usa pseudo-inversa
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            scores[:, c] = exponent + np.log(self.priors[c])
        
        return np.argmax(scores, axis=1) + 1

def monte_carlo_validation(X, Y, model_class, R=500, test_size=0.2, lambda_=None):
    accuracies = []
    n_samples = len(Y)
    
    for _ in range(R):
        indices = np.random.permutation(n_samples)
        X_shuffled, Y_shuffled = X[indices], Y[indices]
        split = int(n_samples * (1 - test_size))
        X_train, Y_train = X_shuffled[:split], Y_shuffled[:split]
        X_test, Y_test = X_shuffled[split:], Y_shuffled[split:]
        
        if lambda_ is not None:
            model = model_class(lambda_)
        else:
            model = model_class()
        
        if isinstance(model, (GaussianClassifierRegularized, GaussianClassifier, GaussianClassifierEqualCov, GaussianClassifierAggregated, NaiveBayesClassifier)):
            model.fit(X_train, Y_train, np.argmax(Y_train, axis=1) + 1)
        else:
            model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        accuracy = np.mean(Y_pred == np.argmax(Y_test, axis=1) + 1)
        accuracies.append(accuracy)
    
    return accuracies

X, Y, labels = load_and_prepare_data()
plot_scatter(X, labels)
   
models = {
    'MQO': MQOClassifier,
    'Gaussian': GaussianClassifier,
    'GaussianEqualCov': GaussianClassifierEqualCov,
    'GaussianAggregated': GaussianClassifierAggregated,
    'NaiveBayes': NaiveBayesClassifier,
    'GaussianRegularized_0': lambda: GaussianClassifierRegularized(0),
    'GaussianRegularized_0.25': lambda: GaussianClassifierRegularized(0.25),
    'GaussianRegularized_0.5': lambda: GaussianClassifierRegularized(0.5),
    'GaussianRegularized_0.75': lambda: GaussianClassifierRegularized(0.75),
    'GaussianRegularized_1': lambda: GaussianClassifierRegularized(1)
}
    
results = {}
for name, model_class in models.items():
    accuracies = monte_carlo_validation(X, Y, model_class)
    results[name] = accuracies
    
for name, accuracies in results.items():
    print(f"{name} - Acurácia média: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    
plt.figure(figsize=(12, 8))
for name, accuracies in results.items():
    plt.hist(accuracies, bins=20, alpha=0.5, label=name)
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição das Acurácias - Validação Monte Carlo')
plt.legend()
