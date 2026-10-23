import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Naive Bayes classifier implemented from scratch
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = []
        self.var = []
        self.prior = []

        for c in self.classes:
            X_c = X[y == c]
            self.mean.append(X_c.mean(axis=0))
            self.var.append(X_c.var(axis=0) + 1e-9)
            self.prior.append(len(X_c) / len(X))

        self.mean = np.array(self.mean)
        self.var = np.array(self.var)
        self.prior = np.array(self.prior)

    def predict(self, X):
        predictions = []

        for sample in X:
            scores = []

            for i in range(len(self.classes)):
                log_likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * self.var[i])
                    + (sample - self.mean[i]) ** 2 / self.var[i]
                )

                score = np.log(self.prior[i]) + log_likelihood
                scores.append(score)

            predictions.append(self.classes[np.argmax(scores)])

        return np.array(predictions)


# Train the classifier
model = NaiveBayes()
model.fit(X_train, y_train)

# Classify one flower
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("Flower measurements:", sample[0])
print("Predicted class:", iris.target_names[prediction[0]])
print("Actual class:", iris.target_names[y_test[0]])
