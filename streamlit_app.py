import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

from clf_viz import viz 

plt.style.use('fivethirtyeight')

st.set_page_config(
    page_title='Ensemble',
    page_icon=":penguin:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None)

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Gaussian Naive Bayes',
    'C-Support Vector Classifier',
    'KNN',
    'Neural Network',
    'Hist-based Gradient Boosting',
    'Adaptive Boosting',
    'QDA'
    ]

data = pd.read_csv('penguins_size.csv')
data = data[
    ['species', 'culmen_length_mm', 'culmen_depth_mm',
    'flipper_length_mm', 'body_mass_g']
    ].dropna()

feature_records = [
        {'name':'culmen_length_mm', 's_name':'Bill Length'},
        {'name':'culmen_depth_mm', 's_name':'Bill Depth'},
        {'name':'flipper_length_mm', 's_name':'Flipper Length'},
        {'name':'body_mass_g', 's_name':'Body Mass'}
]

with st.sidebar:
    st.image(
        "penguins.png",
        use_column_width=True)
    with st.form('Form1'):
        model1 = st.selectbox(
            'Model 1', options=model_names
        )
        w1 = st.slider(
            'Weight 1', 1,5,1
        )

        model_names2 = model_names.copy()
        model_names2.insert(0, model_names2.pop(model_names2.index('Random Forest')))
        model2 = st.selectbox(
            'Model 2', options=model_names2
        )
        w2 = st.slider(
            'Weight 2', 1,5,1
        )

        model_names3 = model_names.copy()
        model_names3.insert(0, model_names3.pop(model_names3.index('KNN')))
        model3 = st.selectbox(
            'Model 3', options=model_names3
        )
        w3 = st.slider(
            'Weight 3', 1,5,1
        )
      
        st.markdown("---")

        feat1_ = st.selectbox(
            'Feature 1', options=feature_records,
            format_func=lambda record: f'{record["s_name"]}'
        )
        feat1 = feat1_.get('name')

        feature_records2 = feature_records.copy()
        feature_records2.insert(0, feature_records2.pop(feature_records2.index({'name':'body_mass_g', 's_name':'Body Mass'})))
        feat2_ = st.selectbox(
            'Feature 2', options=feature_records2,
            format_func=lambda record: f'{record["s_name"]}'
        )
        feat2 = feat2_.get('name')

        weights = [w1,w2,w3]

        submitted = st.form_submit_button('Run')

if not submitted:
    st.markdown(
        """
        ## Ensembling in Machine Learning - Voting Classifiers

        ### This machine learning app provides an interactive way to experiment with\
        <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html" style="color: #b1dcfc">voting classifiers</a>.\
        Using the <a href="https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data" style="color: #b1dcfc">Palmer Penguins dataset</a>,\
        pick three different models and their weights along with two different features and click "Run" to see the results!\
        A decision boundary plot is created along with predicted class probabilities for each point on the plane.\
        Additionally, various scores for each model are generated via 5-fold cross-validation and then averaged.

        #### Can you create a voting classifier that outscores the models that it consists of? Try it out!

        ##### ***Disclaimer:** Please keep in mind that the Palmer Penguins dataset is **toy** data.\
        This modelling approach will not always work in practice. Additionally, although cross validation\
        is used in these examples, it is highly recommended to use a validation set in practice as well.*
        """,
        unsafe_allow_html=True
        )

else:
    if feat1 == feat2 or len(set([model1, model2, model3])) < 3:
        st.error(
            'It looks like you used the same feature or model twice. ' \
            'Please set a different configuration to cheer up Pingu.'
        )
        col1, col2, col3 = st.columns([.00001, 16, .00001])
        with col1:
            st.write(' ')
        with col2:
            st.image('angry_penguin.png',width=900)
        with col3:
            st.write(' ')

        st.stop()

    else:
        st.snow()

        if model1 == 'KNN':
            model1 = (KNeighborsClassifier(n_neighbors=5), 'KNN')
        elif model1 == 'C-Support Vector Classifier':
            model1 = (SVC(probability=True), 'SVC')
        elif model1 == 'Gaussian Naive Bayes':
            model1 = (GaussianNB(), 'Gaussian Naive Bayes')
        elif model1 == 'Random Forest':
            model1 = (RandomForestClassifier(), 'Random Forest')
        elif model1 == 'Decision Tree':
            model1 = (DecisionTreeClassifier(), 'Decision Tree')
        elif model1 == 'Neural Network':
            model1 = (MLPClassifier(alpha=1, max_iter=1000), 'Neural Network')
        elif model1 == 'Hist-based Gradient Boosting':
            model1 = (HistGradientBoostingClassifier(), 'Hist-based Gradient Boosting')
        elif model1 == 'Adaptive Boosting':
            model1 = (AdaBoostClassifier(), 'Adaptive Boosting')
        elif model1 == 'QDA':
            model1 = (QuadraticDiscriminantAnalysis(), 'QDA')
        else:
            model1 = (LogisticRegression(), 'Logistic Regression')

        if model2 == 'KNN':
            model2 = (KNeighborsClassifier(n_neighbors=5), 'KNN')
        elif model2 == 'C-Support Vector Classifier':
            model2 = (SVC(probability=True), 'SVC')
        elif model2 == 'Gaussian Naive Bayes':
            model2 = (GaussianNB(), 'Gaussian Naive Bayes')
        elif model2 == 'Random Forest':
            model2 = (RandomForestClassifier(), 'Random Forest')
        elif model2 == 'Decision Tree':
            model2 = (DecisionTreeClassifier(), 'Decision Tree')
        elif model2 == 'Neural Network':
            model2 = (MLPClassifier(alpha=1, max_iter=1000), 'Neural Network')
        elif model2 == 'Hist-based Gradient Boosting':
            model2 = (HistGradientBoostingClassifier(), 'Hist-based Gradient Boosting')
        elif model2 == 'Adaptive Boosting':
            model2 = (AdaBoostClassifier(), 'Adaptive Boosting')
        elif model2 == 'QDA':
            model2 = (QuadraticDiscriminantAnalysis(), 'QDA')
        else:
            model2 = (LogisticRegression(), 'Logistic Regression')

        if model3 == 'KNN':
            model3 = (KNeighborsClassifier(n_neighbors=5), 'KNN')
        elif model3 == 'C-Support Vector Classifier':
            model3 = (SVC(probability=True), 'SVC')
        elif model3 == 'Gaussian Naive Bayes':
            model3 = (GaussianNB(), 'Gaussian Naive Bayes')
        elif model3 == 'Random Forest':
            model3 = (RandomForestClassifier(), 'Random Forest')
        elif model3 == 'Decision Tree':
            model3 = (DecisionTreeClassifier(), 'Decision Tree')
        elif model3 == 'Neural Network':
            model3 = (MLPClassifier(alpha=1, max_iter=1000), 'Neural Network')
        elif model3 == 'Hist-based Gradient Boosting':
            model3 = (HistGradientBoostingClassifier(), 'Hist-based Gradient Boosting')
        elif model3 == 'Adaptive Boosting':
            model3 = (AdaBoostClassifier(), 'Adaptive Boosting')
        elif model3 == 'QDA':
            model3 = (QuadraticDiscriminantAnalysis(), 'QDA')
        else:
            model3 = (LogisticRegression(), 'Logistic Regression')

        fig, metrics_df = viz(data, model1, model2, model3, feat1, feat2, weights)
        col1, col2, col3 = st.columns([0.25, 2, 0.25])
        with col1:
            st.write('')
        with col2:
            st.pyplot(fig, facecolor='#0E1117', edgecolor='#0E1117')
        with col3:
            st.write('')

        col1, col2, col3 = st.columns([.00001, 16, .00001])
        with col1:
            st.write('')
        with col2:
            st.dataframe(
                (metrics_df
                        .style
                        .background_gradient(cmap='Blues', subset=['accuracy']).set_precision(4)
                        .background_gradient(cmap='Blues_r', subset=['log_loss']).set_precision(4)
                        .background_gradient(cmap='Blues', subset=['precision']).set_precision(4)
                        .background_gradient(cmap='Blues', subset=['recall']).set_precision(4)
                        # .background_gradient(cmap='Blues', subset=['roc_auc']).set_precision(4) # takes too long
                        )
                )
        with col3:
            st.write('')
