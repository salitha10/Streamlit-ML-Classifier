import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA #Feature reduction
import matplotlib.pyplot as plt

import numpy as np
# Header
st.write("""
# Streamlit Machine Learning
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ( "Iris", "Wine", "Breast Cancer"))
# st.sidebar.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select Clasifier", ( "KNN", "SVM", "Random Forest"))

# Load data
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()

    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target

    return x,y

# Call function to get selected dataset
x,y = get_dataset(dataset_name)

# Display data
st.write("shape of dataset", x.shape)
st.write("number of classes", len(np.unique(y)))

#Slider to get classifier
def add_parameter_ui(classifier_name):
    params = dict()

    if classifier_name == "KNN":
        k = st.sidebar.slider("K", 1, 15)
        params["K"] = k
    
    elif classifier_name == "SVM":
        c = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = c
    
    else:
        max_depth = st.sidebar.slider("Max_depth", 2, 15)
        num_of_estimators = st.sidebar.slider("Num_of_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["num_of_estimators"] = num_of_estimators

    return params

#Get classifier
def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors= params["K"])
       
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    
    else:
      clf = RandomForestClassifier(n_estimators=params["num_of_estimators"], max_depth =params["max_depth"], 
                                                                            random_state=1234)
    return clf

clf = get_classifier(classifier_name, add_parameter_ui(classifier_name))

# Clasification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) #split data

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Compute accuracy scores
acc =accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# Plot graph
pca = PCA(2)
x_projected = pca.fit_transform(x)

# Dimensions
x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Proncipal component 1")
plt.ylabel("Proncipal component 2")
plt.colorbar()

# Show graph
st.pyplot(fig)