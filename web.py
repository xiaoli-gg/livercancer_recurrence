#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Prediction model for hepatocellular carcinoma recurrence after hepatectomy: Machine learning-based development and interpretation study')
st.title('Prediction model for hepatocellular carcinoma recurrence after hepatectomy: Machine learning-based development and interpretation study')

#%%set variables selection
st.sidebar.markdown('## Variables')
GGT = st.sidebar.slider("Gamma-glutamyltranspeptidase(Unit/L)", 0, 600, value=30, step=1)
N = st.sidebar.slider("Neutrophil(10^9/L)", 0.00, 12.00, value=1.00, step=0.01)
Fibrinogen = st.sidebar.slider('Fibrinogen(mg/dL)',1.00, 10.00, value=2.00, step=0.01)
Albumin = st.sidebar.slider("Albumin(g/L)", 20.0, 55.0, value=38.0, step=0.1)
TB = st.sidebar.slider("Total bilirubin(μmol/L)", 0.0, 100.0, value=40.0, step=0.1)
M = st.sidebar.slider("Macrophage(10^9/L)", 0.00, 2.00, value=1.00, step=0.01)
AST = st.sidebar.slider("Glutamic oxalacetic transaminase(U/L)", 0, 250, value=50, step=1)
Plt = st.sidebar.slider("Platelet(10^9/L)", 30, 500, value=50, step=1)
ALT = st.sidebar.slider("Glutamic-pyruvic transaminase(U/L)", 1, 400, value=50, step=1)
Total_cholesterol = st.sidebar.slider("Total cholesterol(mmol/L)",0.00,10.00,value = 5.00, step = 0.01)
L = st.sidebar.slider("Lymphocyte(10^9/L)",0.00,5.00,value = 2.50, step =0.01)
NLR = st.sidebar.slider("The ratio of neutrophils to lymphocytes",0.00, 15.00, value =7.50, step = 0.01)
Age = st.sidebar.slider("Age(year)",0,100, value = 40, step = 1)
#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#%%传入数据


#%%load model
mlp_model = joblib.load('mlp_model.pkl')

#%%load data
hp_train = pd.read_csv('liver_cut_data_recurrence_new.csv')
features =["GGT",
            "N",
            "Fibrinogen",
            'Albumin',
            'TB',
            'M',
            'AST',
            'Plt',
            'ALT',
            'Total_cholesterol',
            'L',
            'NLR',
            'Age']
target = "Recurrence"
y = np.array(hp_train[target])
sp = 0.5

is_t = (mlp_model.predict_proba(np.array([[GGT,N,Fibrinogen,Albumin,TB,M,AST,Plt,ALT,Total_cholesterol,L,NLR,Age]]))[0][1])> sp
prob = (mlp_model.predict_proba(np.array([[GGT,N,Fibrinogen,Albumin,TB,M,AST,Plt,ALT,Total_cholesterol,L,NLR,Age]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Recurrence'
else:
    result = 'Low Risk Recurrence'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk recurrence group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[GGT,N,Fibrinogen,Albumin,TB,M,AST,Plt,ALT,Total_cholesterol,L,NLR,Age]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0
    
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = mlp_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of MLP model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of MLP model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of MLP model')
    mlp_prob = mlp_model.predict(X)
    cm = confusion_matrix(y, mlp_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Recurrence', 'Recurrence'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of MLP")
    disp1 = plt.show()
    st.pyplot(disp1)
