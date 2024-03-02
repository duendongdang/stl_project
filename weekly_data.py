import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import plotly.express as px
import altair as alt
import time
import scipy
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

class WeeklyData:
    def __init__(self, data):
        self.data = data
    
    @st.cache_data
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data

    def kmeans_clustering_weekly(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters , random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(self.data[['average_weekly','std_ton_weekly']])
        return self.data 

    def scatter_plot_weekly(self, title, vertical_lower_w, vertical_upper_w, horizontal_lower_w, horizontal_upper_w):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.scatter(self.data['average_weekly'], self.data['std_ton_weekly'], c=self.data['Cluster'], cmap='viridis', s=90)
        ax.set_title(title, fontsize=16 , fontweight='bold',fontfamily='Arial') 
        ax.set_xlabel('average_weekly', fontsize=14)
        ax.set_ylabel('std_ton_weekly', fontsize=14)
        Lower_X_line_W = ax.axhline(y=horizontal_lower_w, color='IndianRed', linestyle='-', linewidth=3)
        Upper_X_line_W = ax.axhline(y=horizontal_upper_w, color='IndianRed', linestyle='-', linewidth=3)
        Lower_Y_line_W = ax.axvline(x=vertical_lower_w, color='IndianRed', linestyle='-', linewidth=3)
        Upper_Y_line_W = ax.axvline(x=vertical_upper_w, color='IndianRed', linestyle='-', linewidth=3)
        ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    def classify_points_weekly(self, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
    # สร้างคอลัมน์ใหม่เพื่อจำแนกพื้นที่
        conditions = [
            (self.data['average_weekly'] >= vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_weekly'] >= vertical_lower) & (self.data['average_weekly'] < vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_upper),
            (self.data['average_weekly'] >= vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_upper),
            (self.data['average_weekly'] < vertical_lower) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_weekly'] >= vertical_lower) & (self.data['average_weekly'] < vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_weekly'] >= vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_weekly'] < vertical_lower) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_weekly'] >= vertical_lower) & (self.data['average_weekly'] < vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_weekly'] >= vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower)
        ]
        choices = range(1, 10)  # ตัวเลข 1 ถึง 9 สำหรับแต่ละพื้นที่
        self.data['Area'] = np.select(conditions, choices, default=0)
        return self.data
        
    
    
