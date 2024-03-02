import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import plotly.express as px
import altair as alt
import time

# file_path = r"C:\stl_project\clustering_average_monthly_std_ton_monthly.xlsx"
# file_path1 = r"C:\stl_project\clustering_average_weekly_std_ton_weekly.xlsx"


@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)

def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters , random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['average_weekly','std_ton_weekly']])
    return data 

def scatter_plot(data, x_axis, y_axis, title):
    fig, ax = plt.subplots()
    ax.scatter(data[x_axis], data[y_axis], c=data['Cluster'], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)
    
st.set_page_config(layout="wide")
st.title('Cluster :green[Analysis] with Streamlit')

uploaded_file = st.file_uploader('Upload Excel File (Dataset 1)', type=['xlsx'])

        
if uploaded_file is not None:
    # อ่านข้อมูลจากไฟล์ Excel
    data_1 = load_data(uploaded_file)

    # ทำ clustering
    n_clusters = st.slider('Select number of clusters', 2, 10, 4)
    clustered_data_1 = kmeans_clustering(data_1, n_clusters)

    st.sidebar.markdown("test choice for chart :chart:")
    x_axis = st.sidebar.selectbox('X-axis', data_1.columns)
    y_axis = st.sidebar.selectbox('Y-axis', data_1.columns ,index = 1)
    
    
    container = st.container()
    chart1, chart2 = container.columns(2)

    with chart1:
        # แสดง scatter plot
        scatter_plot(data=clustered_data_1, x_axis= x_axis,y_axis=y_axis, title='Clusters')


        
    with chart2:
        
        if x_axis in clustered_data_1.columns:
            top5_data = clustered_data_1.sort_values(by=x_axis,ascending=False).head(5)
            
            st.write("Top 5 Data:")
            st.write(top5_data)
            
            labels = top5_data[x_axis].tolist()
            sizes = top5_data[y_axis].tolist()

            # สร้าง pie chart
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  
            st.pyplot(fig) 
        else:
            st.warning("data not found in the dataset.")
            
        