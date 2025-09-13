#!/usr/bin/env python3
# Minimal deep neural network (MLP) in NumPy

import numpy as np

# ----- utilities -----
def one_hot(y, num_classes):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out

def accuracy(y_pred_logits, y_true):
    y_pred = np.argmax(y_pred_logits, axis=1)
    return (y_pred == y_true).mean()

# ----- layers -----
class Dense:
    def __init__(self, in_dim, out_dim, weight_scale=0.01, l2=0.0):
        self.W = np.random.randn(in_dim, out_dim) * weight_scale
        self.b = np.zeros(out_dim)
        self.l2 = l2
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        # grads
        self.dW = self.x.T @ grad_out + self.l2 * self.W
        self.db = grad_out.sum(axis=0)
        # grad wrt input
        return grad_out @ self.W.T

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

# ----- loss (softmax + cross-entropy) -----
class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):  # y_true class indices
        # numeric stability
        z = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(z)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y_true = y_true
        n = logits.shape[0]
        loss = -np.log(self.probs[np.arange(n), y_true] + 1e-12).mean()
        return loss

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.y_true] -= 1.0
        grad /= n
        return grad

# ----- simple MLP model -----
class MLP:
    def __init__(self, dims, l2=0.0):
        """
        dims: list like [in_dim, h1, h2, ..., out_dim]
        """
        self.layers = []
        for i in range(len(dims) - 2):
            self.layers += [Dense(dims[i], dims[i+1], weight_scale=np.sqrt(2/dims[i]), l2=l2), ReLU()]
        self.layers += [Dense(dims[-2], dims[-1], weight_scale=np.sqrt(2/dims[-2]), l2=l2)]
        self.crit = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.step(lr)

    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=64, lr=1e-2, verbose=True):
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            # shuffle
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            # mini-batch SGD
            total_loss = 0.0
            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                logits = self.forward(xb)
                loss = self.crit.forward(logits, yb)
                total_loss += loss * xb.shape[0]

                grad = self.crit.backward()
                self.backward(grad)
                self.step(lr)

            train_loss = total_loss / n
            train_acc = accuracy(self.forward(X), y)

            if verbose:
                msg = f"epoch {epoch:03d} | loss {train_loss:.4f} | acc {train_acc:.4f}"
                if X_val is not None and y_val is not None:
                    val_logits = self.forward(X_val)
                    val_loss = self.crit.forward(val_logits, y_val)
                    val_acc = accuracy(val_logits, y_val)
                    msg += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
                print(msg)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ----- demo on a toy dataset (two moons) -----
if __name__ == "__main__":
    np.random.seed(42)

    # simple synthetic dataset (two circles/ moons without sklearn)
    # generate 2D points in two rings
    def make_two_rings(n_per_class=300, noise=0.08, r1=1.0, r2=2.0):
        t1 = 2*np.pi*np.random.rand(n_per_class)
        t2 = 2*np.pi*np.random.rand(n_per_class)
        x1 = np.c_[r1*np.cos(t1), r1*np.sin(t1)] + noise*np.random.randn(n_per_class, 2)
        x2 = np.c_[r2*np.cos(t2), r2*np.sin(t2)] + noise*np.random.randn(n_per_class, 2)
        X = np.vstack([x1, x2])
        y = np.array([0]*n_per_class + [1]*n_per_class)
        # shuffle
        idx = np.random.permutation(X.shape[0])
        return X[idx], y[idx]

    X, y = make_two_rings(n_per_class=400, noise=0.1)

    # train/val split
    n = X.shape[0]
    val_ratio = 0.2
    n_val = int(n * val_ratio)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    # standardize
    mu, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std

    # build and train: 2 -> 64 -> 64 -> 2
    model = MLP(dims=[2, 64, 64, 2], l2=1e-4)
    model.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=1e-2, verbose=True)

    # evaluate
    logits_val = model.forward(X_val)
    print("final val acc:", accuracy(logits_val, y_val))
