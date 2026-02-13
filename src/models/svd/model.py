import numpy as np


class SVD:
    def __init__(self, n_users, n_items, k=50, lr=0.005, reg=0.02, epochs=20):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # latent factors
        self.U = np.random.normal(0, 0.1, (n_users, k))
        self.V = np.random.normal(0, 0.1, (n_items, k))

        # bias
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        self.global_mean = 0

    def fit(self, train_data):
        self.global_mean = np.mean([r for (_, _, r) in train_data])

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)

            total_loss = 0

            for u, i, r in train_data:
                pred = self.predict_single(u, i)
                error = r - pred

                total_loss += error ** 2

                # update bias
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                # update latent
                self.U[u] += self.lr * (error * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (error * self.U[u] - self.reg * self.V[i])

            rmse = np.sqrt(total_loss / len(train_data))
            print(f"Epoch {epoch+1}/{self.epochs} - RMSE: {rmse:.4f}")

    def predict_single(self, u, i):
        return (
            self.global_mean
            + self.b_u[u]
            + self.b_i[i]
            + np.dot(self.U[u], self.V[i])
        )

    def predict(self, data):
        preds = []
        for u, i, _ in data:
            preds.append(self.predict_single(u, i))
        return np.array(preds)
