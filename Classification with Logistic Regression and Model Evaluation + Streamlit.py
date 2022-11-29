import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

st.title("Logistic Regression")
st.markdown("""
## Upload csv file
""")

uploaded_file = st.file_uploader("Chọn file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    df = "data/" + uploaded_file.name
    with open(df, "wb") as f:
        f.write(bytes_data) 
    dataframe = pd.read_csv(df, sep=';')
    
    st.header("2. Chọn input features: ")
    X = dataframe.iloc[:, :-1]
    for i in X.columns:
        agree = st.checkbox(i)
        if agree == False:
            X = X.drop(i, 1)
    st.write(X)
    
    st.header("4. Output feature")
    y = dataframe.iloc[:, -1]
    st.write(y)

    st.header("5. Chọn tỉ lệ train và test")
    train_data = st.number_input('Insert percentages of train data',min_value=3,max_value=100, value=50)
    st.write('% Train data: ', train_data , '%')
    st.write('% Test data: ', (100 - train_data), '%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - train_data) * 0.01, random_state=0)
    
            
    st.header("6. Sử dụng K-Fold Cross-validation")
    k_fold = st.checkbox('K-Fold Cross-validation')
    if k_fold == True:
        num = st.number_input('Số fold: (runable at 2 and above)')
        num = int(num)

    if st.button('Run'):
        st.write('Logistic Regression')
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train) 
        X_test = sc_x.transform(X_test)
        Lr = LogisticRegression(random_state = 0)
        Lr.fit(X_train, y_train)
        y_pred = Lr.predict(X_test)

        if k_fold == True:
            folds = KFold(n_splits = num, shuffle = True, random_state = 100)
            scores = cross_val_score(Lr, X_train, y_train, scoring='neg_log_loss', cv=folds)
            scores_2 = cross_val_score(Lr, X_train, y_train, scoring='f1', cv=folds)
            df_me = pd.DataFrame(columns = ['Loss', 'F1'])
            for i in range(len(scores)):
                df_me = df_me.append({'Loss' : abs(scores[i]), 'F1' : abs(scores_2[i])}, ignore_index = True)
            st.write(df_me)
            labels = []
            for i in range(num):
                labels.append(str(i+1) + ' Fold')
            X_axis = np.arange(len(labels))
            fig1, ax = plt.subplots(figsize=(20, 20))
            plt.bar(X_axis, df_me['Loss'], width=0.5, color='red', label='Loss')
            plt.xticks(X_axis, labels)
            plt.title('Compare Loss and F1-Score', fontsize=30)
            plt.xlabel('Logistic Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.legend()
            st.pyplot(fig1)

            fig2, ax = plt.subplots(figsize=(20, 20))
            plt.bar(X_axis, df_me['F1'], width=0.5, color='blue', label='F1')
            plt.xticks(X_axis, labels)
            plt.title('Compare Loss and F1-Score', fontsize=30)
            plt.xlabel('Logistic Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.legend()
            st.pyplot(fig2)
        else:
            df_me = pd.DataFrame(columns = ['Loss', 'F1'])
            f1score = f1_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred)
            df_me = df_me.append({'Loss' : logloss, 'F1' : f1score}, ignore_index = True)
            st.write(df_me)
            fig1, ax = plt.subplots(figsize=(20, 20))
            plt.bar(df_me['Loss'], df_me['Loss'], width=0.5, color='red', label='Loss')
            plt.title('Log Loss', fontsize=30)
            plt.xlabel('Logistic Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.legend()
            st.pyplot(fig1)

            fig2, ax = plt.subplots(figsize=(20, 20))
            plt.bar(df_me['F1'], df_me['F1'], width=0.5, color='blue', label='F1')
            plt.title('F1-Score', fontsize=30)
            plt.xlabel('Logistic Regression', fontsize=15)
            plt.ylabel('Metrics', fontsize=15)
            plt.legend()
            st.pyplot(fig2) 
        
