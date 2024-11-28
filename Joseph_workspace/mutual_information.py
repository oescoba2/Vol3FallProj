import numpy as np
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_regression

class FeatureSeparator():
    def __init__(self, data, similarity_metric="cos", alpha=1):
        
        self.data = data

        if similarity_metric not in ["cos", "mi"]:
            raise ValueError("Invalid similarity metric. Use 'cos' for cosine similarity or 'mi' for mutual information.")


        self.similarity_metric = similarity_metric

        if similarity_metric == "cos":
            self.similarity = np.matmul
        else:
            self.similarity = lambda X, Y : mutual_info_regression(X, Y, random_state=3)

        self.alpha = alpha

    def get_separator(self, X, Y):
        X = X.reshape(-1,1) #/ np.linalg.norm(X)
        Y = Y.reshape(-1,1) #/ np.linalg.norm(Y)

        if self.similarity_metric == "cos":
            X = np.ravel(X) 
            Y = np.ravel(Y)

        def obj(D):
            return -np.abs(self.similarity(X, D)) + self.alpha * np.abs(self.similarity(Y, D))

        D = X.reshape(-1)
        D = D / np.linalg.norm(D)
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.linalg.norm(x) - 1}]

        res = minimize(obj, D, constraints=constraints, tol=1e-14)
        
        print("Similarity", self.similarity(X, np.ravel(res.x)))
        print("Similarity", self.similarity(X, np.ravel(X)))
        return res.x

    def get_most_similar_feature(self, D, X, n=None):
        if n is None:
            n = X.shape[1]
        X_arr = np.array(X)
        X_arr = X_arr / np.linalg.norm(X_arr, axis=0)

        if self.similarity_metric == "cos":
            X_arr = X_arr.T

        sim = self.similarity(X_arr, D)
        abs_sim = np.abs(sim)
        keys = np.argsort(abs_sim)[::-1][:n]
        return X.columns[keys], sim[keys]

    def get_separating_feature(self, col1, col2, n=None):
        X = self.data
        D = self.get_separator(X[col1].values, X[col2].values)
        F = X.drop([col1, col2], axis=1)
        return self.get_most_similar_feature(D, F, n)


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('75missing.csv')
    data = data.drop('ISO_A3', axis=1)
    col1 = 'Happiness score'
    col2 = 'NY.GDP.PCAP.PP.KD'

    fs = FeatureSeparator(data, similarity_metric="cos", alpha=5)
    print(fs.get_separating_feature(col1, col2, n=50))

