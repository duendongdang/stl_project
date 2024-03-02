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

class MonthlyData:
    def __init__(self, data):
        self.data = data

    @st.cache_data
    def load_data(file_path):
        data = pd.read_excel(file_path)
        return data

    def kmeans_clustering_monthly(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(self.data[['average_monthly', 'std_ton_monthly']])
        return self.data

    def scatter_plot_monthly(self, title, vertical_lower_m, vertical_upper_m, horizontal_lower_m, horizontal_upper_m):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(self.data['average_monthly'], self.data['std_ton_monthly'], c=self.data['Cluster'], cmap='viridis', s=90)
        ax.set_title(title, fontsize=20, fontweight='bold', fontfamily='Arial')
        ax.set_xlabel('average_monthly')
        ax.set_ylabel('std_ton_monthly')
        ax.axhline(y=horizontal_lower_m, color='IndianRed', linestyle='-', linewidth=3)
        ax.axhline(y=horizontal_upper_m, color='IndianRed', linestyle='-', linewidth=3)
        ax.axvline(x=vertical_lower_m, color='IndianRed', linestyle='-', linewidth=3)
        ax.axvline(x=vertical_upper_m, color='IndianRed', linestyle='-', linewidth=3)
        ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    def classify_points_monthly(self, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
        conditions = [
            (self.data['average_monthly'] >= vertical_upper) & (self.data['std_ton_weekly'] < horizontal_lower),
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
