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

class Box :
    def __init__(self, data):
        self.data = data
    
    @st.cache_data
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data
    
    def kmeans_clustering_9box(self, n_clusters):
        model = KMeans(n_clusters=n_clusters)
        self.data['Cluster'] = model.fit_predict(self.data[['average_monthly', 'std_ton_weekly']])
        return self.data
    
    def scatter_plot_9box(self, title, vertical_lower_w, vertical_upper_w, horizontal_lower_m, horizontal_upper_m):
        fig, ax = plt.subplots(figsize=(14,8))
        ax.scatter(self.data['average_monthly'], self.data['std_ton_weekly'], c=self.data['Cluster'], cmap='viridis',s=90)
        ax.set_title(title,fontsize=20 , fontweight='bold',fontfamily='Arial')
        ax.set_xlabel('Average Monthly ')
        ax.set_ylabel('Std Weekly ')

        # คำนวณขอบเขตของแกน x และ y โดยเพิ่ม margin
        x_margin = (self.data['average_monthly'].max() - self.data['average_monthly'].min()) * 0.1
        y_margin = (self.data['std_ton_weekly'].max() - self.data['std_ton_weekly'].min()) * 0.1
        ax.set_xlim([self.data['average_monthly'].min() - x_margin, self.data['average_monthly'].max() + x_margin])
        ax.set_ylim([self.data['std_ton_weekly'].min() - y_margin, self.data['std_ton_weekly'].max() + y_margin])

        # พล็อตเส้นขอบเขตสีแดง
        Lower_X_line_M = ax.axhline(y=horizontal_lower_m, color='IndianRed', linestyle='-', linewidth=3 )
        Upper_X_line_M = ax.axhline(y=horizontal_upper_m, color='IndianRed', linestyle='-', linewidth=3 )
        Lower_Y_line_W = ax.axvline(x=vertical_lower_w, color='IndianRed', linestyle='-', linewidth=3 )
        Upper_Y_line_W = ax.axvline(x=vertical_upper_w, color='IndianRed', linestyle='-', linewidth=3 )

        ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
    def classify_points_9box(self, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
    # สร้างคอลัมน์ใหม่เพื่อจำแนกพื้นที่
        conditions = [
            (self.data['std_ton_weekly'] >= vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_monthly'] >= vertical_lower) & (self.data['average_monthly'] < vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_upper),
            (self.data['average_monthly'] >= vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_upper),
            (self.data['average_monthly'] < vertical_lower) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_monthly'] >= vertical_lower) & (self.data['average_monthly'] < vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_monthly'] >= vertical_upper) & (self.data['std_ton_weekly'] >= horizontal_lower) & (self.data['std_ton_weekly'] < horizontal_upper),
            (self.data['average_monthly'] < vertical_lower) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_monthly'] >= vertical_lower) & (self.data['average_monthly'] < vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower),
            (self.data['average_monthly'] >= vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower)
        ]
        choices = range(1, 10)  # ตัวเลข 1 ถึง 9 สำหรับแต่ละพื้นที่
        self.data['Area'] = np.select(conditions, choices, default=0)
        return self.data