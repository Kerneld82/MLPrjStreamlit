import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import login, Repository
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import gc

# Baseline
#  - https://www.kaggle.com/code/nguyncngph/instacart-lightgbm
#  - https://www.kaggle.com/code/kokovidis/ml-instacart-f1-0-38-part-one-features)

def getTrainedModel():
    print("getTrainedModel()")
    with open('lightgbm_trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Streamlit 애플리케이션
st.title("Instacart Market Basket Analysis")

if st.button("pickle 파일로부터 학습된 모델 로드 및 예측할 데이터 로드"):
    with st.spinner("학습된 모델을 로드 중입니다..."):
        model = getTrainedModel()
        
        st.session_state.model = model  # 모델을 세션 상태에 저장
        st.write(f"학습된 모델 로드 완료.")
        
        with st.spinner("예측할 데이터를 로드 중입니다.."):
            X_val = pd.read_pickle('X_val.pkl')
            y_val = pd.read_pickle('y_val.pkl')
            
            st.dataframe(X_val)
            st.dataframe(y_val)
            
            st.session_state.X_val = X_val
            st.session_state.y_val = y_val
            
            st.write(f"예측할 데이터 로드 완료.")
        

if st.button("예측"):
    if 'model' in st.session_state and 'X_val' in st.session_state and 'y_val' in st.session_state:
        with st.spinner("예측 중입니다..."):
            X_val = st.session_state.X_val
            y_val = st.session_state.y_val
            
            y_pred = st.session_state.model.predict(X_val)
            
            bin_pred = []

            for pred in y_pred:
                bin_pred.append(1 if pred > 0.22 else 0)

            y_pred = np.array(bin_pred)
            
            accuracy = accuracy_score(y_val, y_pred)
            st.write(f'정확도 : {accuracy}')
            #st.write(classification_report(y_val, y_pred))
            cm = confusion_matrix(y_val, y_pred)
            
            # 시각화
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["0", "1"], yticklabels=["0", "1"])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(plt)
            
    else:
        st.error("먼저 모델과 데이터를 로드해주세요.")