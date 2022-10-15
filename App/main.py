import streamlit as lit
import matplotlib.pyplot as plt
import ml as algo
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd


ml = algo.Main()

lit.markdown("<style>#MainMenu {visibility: hidden;} footer{visibility: hidden;} .css-k3w14i{font-weight: bolder;}</style>", unsafe_allow_html=True)
lit.markdown("<h1 style='color:magenta;' align='center'>WorkSpace</h1>", unsafe_allow_html=True)
lit.write("\n\n")
datasets = lit.selectbox("Choose Your Dataset", options=["none", "iris", "linnerud"])

def iris(test_size):
    ml.dataset(ml.iris.data, ml.iris.feature_names)
    ml.data["target"] = ml.iris.target
    X, y = ml.values(x=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ], y="target")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def linnerud(test_size):
    ml.dataset(ml.excercise.data, ml.excercise.feature_names)
    # ml.data[['Weight', 'Waist', 'Pulse']] = ml.excercise.target
    X, y = ml.values(x=[
        "Chins",
        "Situps"
    ], y="Jumps")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def pca(X_train, y_train):
    if lit.sidebar.checkbox("PCA", key="PCA1"):
        solver = lit.sidebar.selectbox("Solver", ["full", "arpack"])
        vria = lit.sidebar.selectbox("Values", ["Singular Values", "Explained Variance ratio"])
        # singular_values_, explained_variance_ratio_
        principal = PCA(n_components=1, svd_solver=solver).fit(X_train, y_train)
        if vria == "Singular Values":
            value = principal.singular_values_
        elif vria == "Explained Variance ratio":
            value = principal.explained_variance_ratio_
        else:
            value = "Please Pick Any Value Decider"
        return value

def KFold(X, y):
    if lit.sidebar.checkbox("K-Fold", key="key_fold"):
        models = lit.sidebar.selectbox("Models Type", ["LinearRegression", "LogisticRegressions", "SVC", "RandomForestClassifier", "KNeighborsClassifier"])
        cv = lit.sidebar.slider("Fold", 2,7,2)
        if models == "LinearRegression":
            values = ml.folding(LinearRegression(), X=X, y=y, cv=cv)
        elif models == "LogisticRegressions":
            values = ml.folding(LogisticRegression(), X=X, y=y, cv=cv)
        elif models == "SVC":
            values = ml.folding(SVC(), X, y, cv)
        elif models == "RandomForestClassifier":
            values = ml.folding(RandomForestClassifier(), X, y, cv)
        elif models == "KNeighborsClassifier":
            values = ml.folding(KNeighborsClassifier(), X, y, cv)
        else:
            values = "Please Pick Any Model"
        return values

def plot_iris():
    iris = ml.iris     # load_iris()
    dx = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = dx["sepal length (cm)"]
    y = dx["sepal width (cm)"]
    x1 = dx["petal length (cm)"]
    y1 = dx["petal width (cm)"]
    if lit.checkbox("Data Plotting"):
        box = lit.selectbox("Select Your Graph", ["All", "Sepla Length & Width (cm)", "Petal Length & Width (cm)"])
        if box == "All":
            figure = plt.figure()
            ax = plt.axes()
            ax.set_facecolor("#262730")
            plt.scatter(x.values, y.values)
            plt.xlabel("sepal length (cm)")
            plt.ylabel("sepal width (cm)")
            lit.pyplot(figure)
            figure1 = plt.figure()
            ax1 = plt.axes()
            ax1.set_facecolor("#262730")
            plt.scatter(x1.values, y1.values, c="red")
            plt.xlabel("Petal length (cm)")
            plt.ylabel("Petal width (cm)")
            lit.pyplot(figure1)
            figure2 = plt.figure()
            ax2 = plt.axes()
            ax2.set_facecolor("#262730")
            plt.scatter(dx[["petal length (cm)"]].values, dx[["petal width (cm)"]].values, c="yellow")
            plt.scatter(dx[["sepal length (cm)"]].values, dx[["sepal width (cm)"]].values, c="green")
            plt.xlabel("Length (cm)")
            plt.ylabel("Width (cm)")
            lit.pyplot(figure2)

        elif box == "Sepla Length & Width (cm)":
            figure = plt.figure()
            ax = plt.axes()
            ax.set_facecolor("#262730")
            plt.scatter(x.values, y.values)
            plt.xlabel("sepal length (cm)")
            plt.ylabel("sepal width (cm)")
            lit.pyplot(figure)

        elif box == "Petal Length & Width (cm)":
            figure1 = plt.figure()
            ax1 = plt.axes()
            ax1.set_facecolor("#262730")
            plt.scatter(x1.values, y1.values, c="red")
            plt.xlabel("Petal length (cm)")
            plt.ylabel("Petal width (cm)")
            lit.pyplot(figure1)

        else:
            print("Please Select Any Graph Type")

def plot_linnerud():
    linnerud = ml.excercise
    df = pd.DataFrame(linnerud.data, columns=linnerud.feature_names)

    if lit.checkbox("Data Plotting"):
        box = lit.selectbox("Select Your Graph", ["All", "Chins & Situps"])
        if box == "All":
            fig = plt.figure()
            ax = plt.axes()
            ax.set_facecolor("#262730")
            plt.scatter(df[["Chins"]], df["Situps"])
            lit.pyplot(fig)

            fig2 = plt.figure()
            ax2 = plt.axes()
            ax2.set_facecolor("#262730")
            plt.scatter(df[["Jumps"]], df["Situps"])
            plt.scatter(df[["Jumps"]], df["Chins"])
            lit.pyplot(fig2)

        elif box == "Chins & Situps":
            fig = plt.figure()
            ax = plt.axes()
            ax.set_facecolor("#262730")
            plt.scatter(df[["Chins"]], df["Situps"])
            lit.pyplot(fig)

        else:
            print("Please Select Any Graph")

def models(X, y):
    scaler = lit.radio("Scale", ["False", "True"])
    if scaler == "True":
        clf = algo.Classifier(scale=True)
        regressor = algo.Regression(scale=True)
        model = ["LinearRegression", "LogisticRegressions", "SVC", "RandomForestClassifier", "KNeighborsClassifier"]
        box = lit.selectbox("Choose Your Classifier", model)
        if box == model[0]:
            predict = regressor.LinearRegression(X, y)
        elif box == model[1]:
            iters = lit.slider("Max iter", 1, 1000, 100)
            predict = clf.LogisticRegression(iters=iters, X_train=X, y_train=y)
        elif box == model[2]:
            c = lit.slider("Support Vector Classification", 1, 100, 10)
            gamma = lit.select_slider("Gamma", ["scale", "auto"])
            kernel = lit.selectbox("Choose Your SVC Kernel", ['rbf', 'poly', 'linear', 'sigmoid'])
            predict = clf.SVC(X, y, c, gamma=gamma, kernel=kernel)
        elif box == model[3]:
            tress = lit.slider("Forest Tress", 1, 100, 40)
            predict = clf.RandomForest(tress, X, y)
        elif box == model[4]:
            k = lit.slider("K Value", 1, 10, 3)
            predict = clf.KNeighborsClassifier(k, X, y)
        else:
            predict = ""
        predicted = predict


    elif scaler == "False":
        clf = algo.Classifier(scale=False)
        regressor = algo.Regression(scale=True)
        model = ["LinearRegression", "LogisticRegressions", "SVC", "RandomForestClassifier", "KNeighborsClassifier"]
        box = lit.selectbox("Choose Your Classifier", model)
        if box == model[0]:
            predict = regressor.LinearRegression(X, y)
        elif box == model[1]:
            iters = lit.slider("Max iter", 1, 1000, 100)
            predict = clf.LogisticRegression(iters=iters, X_train=X, y_train=y)
        elif box == model[2]:
            c = lit.slider("Support Vector Classification", 1, 100, 10)
            gamma = lit.select_slider("Gamma", ["scale", "auto"])
            kernel = lit.selectbox("Choose Your SVC Kernel", ['rbf', 'poly', 'linear', 'sigmoid'])
            predict = clf.SVC(X, y, c, gamma=gamma, kernel=kernel)
        elif box == model[3]:
            tress = lit.slider("Forest Tress", 1, 100, 40)
            predict = clf.RandomForest(tress, X, y)
        elif box == model[4]:
            k = lit.slider("K Value", 1, 10, 3)
            predict = clf.KNeighborsClassifier(k, X, y)
        else:
            predict = ""
        predicted = predict

    else:
        predicted = ""
        print("Return Your Button")

    return predicted

def flower(value:int):
    if value == 0:
        name = "setosa"
    elif value == 1:
        name = "versicolor"
    elif value == 2:
        name = "virginica"
    else:
        name = "not Found"

    return name

def iris_input():
    """
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    """
    sepal_length = lit.number_input("Sepal Length (cm)")
    sepal_width = lit.number_input("Sepal Width (cm)")
    petal_len = lit.number_input("Petal Length (cm)")
    petal_width = lit.number_input("Petal Width (cm)")

    return sepal_length, sepal_width, petal_len, petal_width

def lennud():
    """
    Chins
    Situps
    Jumps
    """
    chins = lit.number_input("Chins")
    Situps = lit.number_input("Situps")

    return chins, Situps

lit.sidebar.markdown("<h3 style='color:magenta;'>Analysing Features</h3>", unsafe_allow_html=True)
if datasets == "iris":
    lit.write("\n\n")
    test_size = lit.slider("Select the Test Size of the Model", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = iris(test_size=test_size)
    lit.sidebar.write("")
    kfold = KFold(X_train, y_train)
    lit.sidebar.write(str(kfold))
    pca = pca(X_train=X_train, y_train= y_train)
    if pca == None:
        lit.sidebar.write(str(" "))
    else:
        lit.sidebar.write(str(pca))
    plot_iris()
    model = models(X_train, y_train)
    if lit.checkbox("Score"):
        score = str(np.round(model.score(X_test, y_test)*100, 1))
        lit.success(f"Your Model Score is {score}%")
    sepal_length, sepal_width, petal_len, petal_width = iris_input()
    if lit.button("Predict"):
        suc = lit.success(f"Flower Species is '{flower(int(model.predict([[sepal_length, sepal_width, petal_len, petal_width]])))}'")


elif datasets == "linnerud":
    # lit.markdown("<h3 style='color:rgb(46,123,175);'>Training</h3>", unsafe_allow_html=True)
    lit.write("\n\n")
    test_size = lit.slider("Select the Test Size of the Model", 0.1, 0.5, 0.2)
    # Random_State = lit.slider("Select the Test Size of the Model", 0.0, 1.0, 0.03)
    X_train, X_test, y_train, y_test = linnerud(test_size=test_size)
    lit.sidebar.write("")
    kfold = KFold(X_train, y_train)
    lit.sidebar.write(str(kfold))
    pca = pca(X_train=X_train, y_train= y_train)
    if pca == None:
        lit.sidebar.write(" ")
    else:
        lit.sidebar.write(str(pca))
    plot_linnerud()
    model = models(X_train, y_train)
    if lit.checkbox("Score"):
        score = str(np.round(model.score(X_test, y_test)*100, 1))
        lit.success(f"Your Model Score is {score}%")
    chins, Situps = lennud()
    if lit.button("Predict"):
        lit.success(model.predict([[chins, Situps]]))

else:
    lit.warning("Please Select Any DataSet")



