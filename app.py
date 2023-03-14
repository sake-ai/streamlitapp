import streamlit as st
from sklearn.datasets import load_breast_cancer
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Classifiers in Action")

# Description
st.text("Breast Cancer Dataset")

#sidebar
sideBar = st.sidebar
classifier = sideBar.selectbox('Which Classifier do you want to use?',('SVM' , 'KNN' , 'Random Forest'))
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names , index=None)
df['Type'] = data.target
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=1, test_size=0.2)


classes = data.target_names
st.dataframe(df.sample(n = 5 , random_state = 1))
st.subheader("Classes")


for idx, value in enumerate(classes):
    st.text('{}: {}'.format(idx , value))


if classifier == 'SVM':
        c = st.sidebar.slider(label='Choose value of C' , min_value=0.0001, max_value=10.0)
        model = SVC(C=c)
        model.fit(X_train, y_train)
        test_score = round(model.score(X_test, y_test), 2)
        train_score = round(model.score(X_train, y_train), 2)
        accuracy=test_score
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=classes).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=classes).round(2))
        # The below code will plot confusion matrix for SVM
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

elif classifier == 'KNN':
        neighbors = st.sidebar.slider(label='Choose Number of Neighbors',min_value=1,max_value=20)
        model = KNeighborsClassifier(n_neighbors = neighbors)
        model.fit(X_train, y_train)
        test_score = round(model.score(X_test, y_test), 2)
        train_score = round(model.score(X_train, y_train), 2)
        accuracy=test_score
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=classes).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=classes).round(2))
        # The below code will plot confusion matrix for KNN
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

else:
    max_depth = st.sidebar.slider('max_depth', 2, 10)
    n_estimators = st.sidebar.slider('n_estimators', 1, 100)
    model = RandomForestClassifier(max_depth = max_depth , n_estimators= n_estimators,random_state= 1)
    model.fit(X_train, y_train)
    test_score = round(model.score(X_test, y_test), 2)
    train_score = round(model.score(X_train, y_train), 2)
    accuracy=test_score
    y_pred = model.predict(X_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_pred, labels=classes).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=classes).round(2))
    # The below code will plot confusion matrix for Random Forest
    st.subheader("ROC Curve")
    plot_roc_curve(model, X_test, y_test)
    st.pyplot()





