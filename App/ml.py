import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_linnerud


class Basic():
    def basic_config(self):
        logging.basicConfig(
            filename=self.log_filename,
            format='%(asctime)s %(message)s',
            filemode="w"
        )

        self.logging = logging.getLogger()
        self.logging.setLevel(logging.DEBUG)

    def __init__(self,  filename: str = "model.log", n_components: int = 1, svd_solver: str = 'arpack') -> None:
        self.log_filename = filename
        self.n_components = n_components
        self.solver_svd = svd_solver
        self.iris = load_iris()
        self.excercise = load_linnerud()

    def dataset(self, data, col_name):
        try:
            self.data = pd.DataFrame(data, columns=col_name)
        except Exception and ValueError:
            self.logging.critical("Failed To load")

    def values(self, x:list, y:str):
        self.X = self.data[x].values
        self.Y = self.data[y].values
        return self.X, self.Y

    def trainer(self, x, y, test_size, random_state: float = 0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, random_state=random_state,  test_size=test_size)

    def PCA(self, n_components, solver_svd, X_train, y_train):
        principal = PCA(n_components=n_components,
                             svd_solver=solver_svd)
        principal.fit(X_train, y_train)
        return principal

    def folding(self, model, X, y, cv: int):
        return cross_val_score(model, X, y, cv=cv)


class Classifier(Basic):
    def __init__(self, filename: str = "model.log", n_components: int = 1, svd_solver: str = 'arpack', scale: bool = False) -> None:
        super().__init__(filename, n_components, svd_solver)
        self.scale = scale

    def pipeline(self, scales, model):
        return make_pipeline(scales, model)

    def LogisticRegression(self, iters: int, X_train, y_train):
        if self.scale == True:
            LogisticRegressions = self.pipeline(
                StandardScaler(), LogisticRegression(max_iter=iters))
            fitted = LogisticRegressions.fit(X_train, y_train)
        else:
            LogisticRegressions = LogisticRegression(max_iter=iters)
            fitted = LogisticRegressions.fit(X_train, y_train)
        return fitted

    def SVC(self,X_train, y_train, C: int, gamma:str, kernel: str):
        if self.scale == True:
            svm = self.pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma=gamma))
            fitted = svm.fit(X_train, y_train)

        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma)
            fitted = svm.fit(X_train, y_train)

        return fitted

    def RandomForest(self, n_estimators: int, X_train, y_train):
        if self.scale == True:
            rand = self.pipeline(
                StandardScaler(), RandomForestClassifier(n_estimators=n_estimators))
            fitted = rand.fit(X_train, y_train)
        else:
            rand = RandomForestClassifier(n_estimators=n_estimators)
            fitted = rand.fit(X_train, y_train)

        return fitted

    def KNeighborsClassifier(self, k: int, X_train, y_train):
        if self.scale == True:
            KNN = self.pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=k))
            fitted = KNN.fit(X_train, y_train)
        else:
            KNN = KNeighborsClassifier(n_neighbors=k)
            fitted = KNN.fit(X_train, y_train)

        return fitted


class Regression(Classifier):
    def LinearRegression(self, X_train, y_train):
        if self.scale == True:
            lr = self.pipeline(StandardScaler(), LinearRegression())
            fitted = lr.fit(X_train, y_train)
        else:
            lr = LinearRegression()
            fitted = lr.fit(X_train, y_train)
        return fitted

    def save(self, model, filename: str):
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(model, f)

class Main(Regression):
    """
    :TODO

    it is conating all App classes

    """
