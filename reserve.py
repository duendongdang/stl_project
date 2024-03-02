
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np
import plotly.express as px
import altair as alt
import time
from scipy import stats
import plotly.graph_objects as go
from streamlit_option_menu import option_menu


@st.cache_data
def load_data(file_path):
    file_name = file_path.name
    if file_name.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_name.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format")
    return data

def kmeans_clustering_weekly(data_w, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters , random_state=42)
    data_w['Cluster'] = kmeans.fit_predict(data_w[['average_weekly','std_ton_weekly']])
    return data_w 

def kmeans_clustering_monthly(data_m, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters , random_state=42)
    data_m['Cluster'] = kmeans.fit_predict(data_m[['average_monthly','std_ton_monthly']])
    return data_m 

def kmeans_clustering_9box(data_9box, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    data_9box['Cluster'] = model.fit_predict(data_9box[['average_monthly', 'std_ton_weekly']])
    return data_9box

def scatter_plot_weekly(data_w, title, vertical_lower_w, vertical_upper_w, horizontal_lower_w, horizontal_upper_w):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(data_w['average_weekly'], data_w['std_ton_weekly'], c=data_w['Cluster'], cmap='viridis', s=90)
    ax.set_title(title, fontsize=16 , fontweight='bold',fontfamily='Arial') 
    ax.set_xlabel('average_weekly', fontsize=14)
    ax.set_ylabel('std_ton_weekly', fontsize=14)
    Lower_X_line_W = ax.axhline(y=horizontal_lower_w, color='IndianRed', linestyle='-', linewidth=3)
    Upper_X_line_W = ax.axhline(y=horizontal_upper_w, color='IndianRed', linestyle='-', linewidth=3)
    Lower_Y_line_W = ax.axvline(x=vertical_lower_w, color='IndianRed', linestyle='-', linewidth=3)
    Upper_Y_line_W = ax.axvline(x=vertical_upper_w, color='IndianRed', linestyle='-', linewidth=3)
    ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
def scatter_plot_monthly(data_m, title, vertical_lower_m, vertical_upper_m, horizontal_lower_m, horizontal_upper_m):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(data_m['average_monthly'], data_m['std_ton_monthly'], c=data_m['Cluster'], cmap='viridis',s=90)
    ax.set_title(title,fontsize=20 , fontweight='bold',fontfamily='Arial')
    ax.set_xlabel('average_monthly')
    ax.set_ylabel('std_ton_monthly')
    Lower_X_line_M = ax.axhline(y=horizontal_lower_m, color='IndianRed', linestyle='-', linewidth=3)
    Upper_X_line_M = ax.axhline(y=horizontal_upper_m, color='IndianRed', linestyle='-', linewidth=3)
    Lower_Y_line_M = ax.axvline(x=vertical_lower_m, color='IndianRed', linestyle='-', linewidth=3)
    Upper_Y_line_M = ax.axvline(x=vertical_upper_m, color='IndianRed', linestyle='-', linewidth=3)
    ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
def scatter_plot_9box(data_9box, title, vertical_lower_w, vertical_upper_w, horizontal_lower_m, horizontal_upper_m):
    fig, ax = plt.subplots(figsize=(14,8))
    ax.scatter(data_9box['average_monthly'], data_9box['std_ton_weekly'], c=data_9box['Cluster'], cmap='viridis',s=90)
    ax.set_title(title,fontsize=20 , fontweight='bold',fontfamily='Arial')
    ax.set_xlabel('Average Monthly ')
    ax.set_ylabel('Std Weekly ')

    # คำนวณขอบเขตของแกน x และ y โดยเพิ่ม margin
    x_margin = (data_9box['average_monthly'].max() - data_9box['average_monthly'].min()) * 0.1
    y_margin = (data_9box['std_ton_weekly'].max() - data_9box['std_ton_weekly'].min()) * 0.1
    ax.set_xlim([data_9box['average_monthly'].min() - x_margin, data_9box['average_monthly'].max() + x_margin])
    ax.set_ylim([data_9box['std_ton_weekly'].min() - y_margin, data_9box['std_ton_weekly'].max() + y_margin])

    # พล็อตเส้นขอบเขตสีแดง
    Lower_X_line_M = ax.axhline(y=horizontal_lower_m, color='IndianRed', linestyle='-', linewidth=3 )
    Upper_X_line_M = ax.axhline(y=horizontal_upper_m, color='IndianRed', linestyle='-', linewidth=3 )
    Lower_Y_line_W = ax.axvline(x=vertical_lower_w, color='IndianRed', linestyle='-', linewidth=3 )
    Upper_Y_line_W = ax.axvline(x=vertical_upper_w, color='IndianRed', linestyle='-', linewidth=3 )

    ax.grid(True, linewidth=1, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    
    
def classify_points_weekly(data, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
    # สร้างคอลัมน์ใหม่เพื่อจำแนกพื้นที่
    conditions = [
        (data['average_weekly'] < vertical_lower) & (data['std_ton_weekly'] >= horizontal_upper),
        (data['average_weekly'] >= vertical_lower) & (data['average_weekly'] < vertical_upper) & (data['std_ton_weekly'] >= horizontal_upper),
        (data['average_weekly'] >= vertical_upper) & (data['std_ton_weekly'] >= horizontal_upper),
        (data['average_weekly'] < vertical_lower) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_weekly'] >= vertical_lower) & (data['average_weekly'] < vertical_upper) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_weekly'] >= vertical_upper) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_weekly'] < vertical_lower) & (data['std_ton_weekly'] < horizontal_lower),
        (data['average_weekly'] >= vertical_lower) & (data['average_weekly'] < vertical_upper) & (data['std_ton_weekly'] < horizontal_lower),
        (data['average_weekly'] >= vertical_upper) & (data['std_ton_weekly'] < horizontal_lower)
    ]
    choices = range(1, 10)  # ตัวเลข 1 ถึง 9 สำหรับแต่ละพื้นที่
    data['Area'] = np.select(conditions, choices, default=0)
    return data

def classify_points_monthly(data, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
    # สร้างคอลัมน์ใหม่เพื่อจำแนกพื้นที่
    conditions = [
        (data['average_monthly'] < vertical_lower) & (data['std_ton_monthly'] >= horizontal_upper),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_monthly'] >= horizontal_upper),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_monthly'] >= horizontal_upper),
        (data['average_monthly'] < vertical_lower) & (data['std_ton_monthly'] >= horizontal_lower) & (data['std_ton_monthly'] < horizontal_upper),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_monthly'] >= horizontal_lower) & (data['std_ton_monthly'] < horizontal_upper),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_monthly'] >= horizontal_lower) & (data['std_ton_monthly'] < horizontal_upper),
        (data['average_monthly'] < vertical_lower) & (data['std_ton_monthly'] < horizontal_lower),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_monthly'] < horizontal_lower),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_monthly'] < horizontal_lower)
    ]
    choices = range(1, 10)  # ตัวเลข 1 ถึง 9 สำหรับแต่ละพื้นที่
    data['Area'] = np.select(conditions, choices, default=0)
    return data

def classify_points_9box(data, vertical_lower, vertical_upper, horizontal_lower, horizontal_upper):
    # สร้างคอลัมน์ใหม่เพื่อจำแนกพื้นที่
    conditions = [
        (data['std_ton_weekly'] >= vertical_upper) & (data['std_ton_weekly'] < horizontal_lower),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_weekly'] >= horizontal_upper),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_weekly'] >= horizontal_upper),
        (data['average_monthly'] < vertical_lower) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_weekly'] >= horizontal_lower) & (data['std_ton_weekly'] < horizontal_upper),
        (data['average_monthly'] < vertical_lower) & (data['std_ton_weekly'] < horizontal_lower),
        (data['average_monthly'] >= vertical_lower) & (data['average_monthly'] < vertical_upper) & (data['std_ton_weekly'] < horizontal_lower),
        (data['average_monthly'] >= vertical_upper) & (data['std_ton_weekly'] < horizontal_lower)
    ]
    choices = range(1, 10)  # ตัวเลข 1 ถึง 9 สำหรับแต่ละพื้นที่
    data['Area'] = np.select(conditions, choices, default=0)
    return data


def main():
    st.set_page_config(page_title='Dashboard for data analysis', page_icon='📉', layout="wide")
    st.sidebar.image("data.png", caption="Dashboard for data analysis")
    
    # Initialize session state variables for uploaded files if not already present
    if 'uploaded_file_weekly' not in st.session_state:
        st.session_state['uploaded_file_weekly'] = None
    if 'uploaded_file_monthly' not in st.session_state:
        st.session_state['uploaded_file_monthly'] = None
    if 'uploaded_file_9box' not in st.session_state:
        st.session_state['uploaded_file_9box'] = None
    if 'uploaded_file_calculate' not in st.session_state:
        st.session_state['uploaded_file_calculate'] = None

    with st.sidebar:
        selected = option_menu("Main Menu", ["Weekly Data", "Monthly Data", "9-Box", "Calculate"], 
                               icons=["calendar-week", "calendar-month", "box", "calculator"], default_index=0)

    if selected in ["Weekly Data", "Monthly Data", "9-Box"]:
        st.title('Cluster :green[Analysis] with Streamlit📊📉')
        uploaded_file = st.file_uploader('Upload File', type=['xlsx','csv'])

        if uploaded_file is not None:
            if selected == "Weekly Data":
                st.session_state['uploaded_file_weekly'] = uploaded_file
            elif selected == "Monthly Data":
                st.session_state['uploaded_file_monthly'] = uploaded_file
            elif selected == "9-Box":
                st.session_state['uploaded_file_9box'] = uploaded_file
        data_1 = load_data(uploaded_file)
        

    if selected == "Weekly Data":
            input_col1 , input_col2 = st.columns([1,2])
            with input_col1:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('##### Input Number In TextBox⬇️')
                horizontal_min = round(float(data_1['std_ton_weekly'].min()),2)
                horizontal_max = round(float(data_1['std_ton_weekly'].max()),2)
                vertical_min = round(float(data_1['average_weekly'].min()),2)
                vertical_max = round(float(data_1['average_weekly'].max()),2)
                    
                # รับค่าเส้นแนวนอนล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
                horizontal_Lower_W = st.text_input('Lower X line (weekly)', value=str(round(horizontal_min, 2)))
                try:
                    horizontal_Lower_W = float(horizontal_Lower_W)
                except ValueError:
                    horizontal_Lower_W = horizontal_min
                    horizontal_Lower_W = round(horizontal_Lower_W, 2)
                    
                    # รับค่าเส้นแนวนอนบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
                horizontal_Upper_W = st.text_input('Upper X line (weekly)', value=str(round(horizontal_max, 2)))
                try:
                    horizontal_Upper_W = float(horizontal_Upper_W)
                except ValueError:
                    horizontal_Upper_W = horizontal_max
                    horizontal_Upper_W = round(horizontal_Upper_W, 2)

                    # รับค่าเส้นแนวตั้งล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
                vertical_Lower_W = st.text_input('Lower Y line (weekly)', value=str(round(vertical_min, 2)))
                try:
                    vertical_Lower_W = float(vertical_Lower_W)
                except ValueError:
                    vertical_Lower_W = vertical_min
                    vertical_Lower_W = round(vertical_Lower_W, 2)
                    
                    # รับค่าเส้นแนวตั้งบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
                vertical_Upper_W = st.text_input('Upper Y line (weekly)', value=str(round(vertical_max, 2)))
                try:
                    vertical_Upper_W = float(vertical_Upper_W)
                except ValueError:
                    vertical_Upper_W = vertical_max
                    vertical_Upper_W = round(vertical_Upper_W, 2)
                    
                    # ตรวจสอบว่าค่าที่ใส่มาอยู่ในช่วงที่ถูกต้องหรือไม่
                if horizontal_Lower_W >= horizontal_Upper_W or vertical_Lower_W >= vertical_Upper_W:
                    st.error("ค่า Lower ต้องน้อยกว่าค่า Upper.")
                    st.stop()

                # คำนวณข้อมูลที่จำแนกตามพื้นที่
                data_classified = classify_points_weekly(data_1, vertical_Lower_W, vertical_Upper_W, horizontal_Lower_W, horizontal_Upper_W)
            
            with input_col2:
                    n_clusters_weekly = st.slider('Select number of clusters for weekly', 2, 10, 3)
                    clustered_data_weekly = kmeans_clustering_weekly(data_classified, n_clusters_weekly)
                    scatter_plot_weekly(data_w=clustered_data_weekly, title='Clusters Weekly',
                                        vertical_lower_w=vertical_Lower_W,
                                        vertical_upper_w=vertical_Upper_W,
                                        horizontal_lower_w=horizontal_Lower_W,
                                        horizontal_upper_w=horizontal_Upper_W )
                    # area_counts_weekly = clustered_data_weekly['Area'].value_counts()
                    
            

            st.markdown('<h4 style="text-align:center;">🔍Clustered Data - Weekly</h4>', unsafe_allow_html=True)
            st.markdown('')
            st.dataframe(clustered_data_weekly[['Product','Grade','Gram','average_weekly', 'std_ton_weekly', 'Cluster', 'Area']],width=1500, height=300)
            st.markdown('')
            st.markdown('')
            st.markdown('<h4 style="text-align:center;">🔢Count Tonnage and Product From 9-Box</h4>', unsafe_allow_html=True)
            st.markdown("""
                            <style>
                            .metric-box {
                                border: 1px solid #e1e4e8; /*สีขอบกรอบ*/
                                background-color: #FFCAC8; /*เปลี่ยนสีพื้นหลังกรอบ*/
                                border-radius: 5px; 
                                padding: 5px 10px;
                                margin: 10px 0;
                                text-align: center;
                                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                            }
                            .metric-box h2 {
                                font-size: 1rem;
                                color: #888; /* เปลี่ยนสีข้อความหัว */
                                margin: 0;
                            }
                            .metric-box h1 {
                                font-size: 2.5rem;
                                color: #000;
                                margin: 0;
                            }
                            </style>
                        """, unsafe_allow_html=True)

            first_row_cols = st.columns(3)
            second_row_cols = st.columns(3) 
            cluster_type={
                0:'non-std', 1:'low-std',2:'std'
            }
            area_type = {
                1: 'std', 2: 'std', 3: 'low-std', 
                4: 'std', 5: 'low-std', 6: 'non-std', 
                7: 'low-std', 8: 'non-std', 9: 'non-std'
            }
            
            data_1['Product Type'] = data_1['Area'].map(area_type)
            data_1['Product Type Cluster'] = data_1['Cluster'].map(cluster_type)
           
            grouped_data = data_1.groupby('Product Type').agg(
                Total_Ton=('ton', 'sum'),  
                Product_Count=('Product', 'count') 
            ).reset_index()
            
            # grouped_data
            first_row_cols = st.columns(3)
            for i, category in enumerate(['std', 'low-std', 'non-std']):
                if category in grouped_data['Product Type'].values:
                    total_ton = grouped_data.loc[grouped_data['Product Type'] == category, 'Total_Ton'].values[0]
                    formatted_total_ton = "{:,.3f}".format(total_ton)
                else:
                    formatted_total_ton = "0.000"
                with first_row_cols[i]:
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Total Ton ({category})</h2>
                        <h1>{formatted_total_ton}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
            second_row_cols = st.columns(3)
            for i, category in enumerate(['std', 'low-std', 'non-std']):
                if category in grouped_data['Product Type'].values:
                    product_count = grouped_data.loc[grouped_data['Product Type']== category, 'Product_Count'].values[0]
                else:
                    product_count  ="0"
                with second_row_cols[i]:
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Product Count ({category})</h2>
                        <h1>{product_count}</h1>
                        </div>
                        
                        """, unsafe_allow_html=True)
            st.markdown('')
            st.markdown('')
          
            

            st.markdown('<h4 style="text-align:center;">🧮Dataframe</h4>', unsafe_allow_html=True)
            st.markdown('')
            st.markdown('')
            st.dataframe(data_1[['Product','Grade','Gram','ton','number_of_week','average_weekly','std_ton_weekly','Product Type','Product Type Cluster' ]],width=1500, height=300)
            st.stop()
        
    # ✋🏻 ---------------------------------------------------------------------------------------------------------------------------        
        
    elif selected == "Monthly Data":
        input_col1,input_col2 = st.columns([1,2])
        with input_col1:
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('##### Input Number In TextBox⬇️')
            # ใช้ข้อมูลที่มีอยู่เพื่อกำหนดค่าเริ่มต้นให้กับช่องใส่ข้อมูล
            horizontal_min_m = round(float(data_1['std_ton_monthly'].min()), 2)
            horizontal_max_m = round(float(data_1['std_ton_monthly'].max()), 2)
            vertical_min_m = round(float(data_1['average_monthly'].min()), 2)
            vertical_max_m = round(float(data_1['average_monthly'].max()), 2)

            # รับค่าเส้นแนวนอนล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            horizontal_Lower_M = st.text_input('Lower X line (monthly)', value=str(horizontal_min_m))
            try:
                horizontal_Lower_M = float(horizontal_Lower_M)
            except ValueError:
                horizontal_Lower_M = horizontal_min_m
                horizontal_Lower_M = round(horizontal_Lower_M, 2)

            # รับค่าเส้นแนวนอนบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            horizontal_Upper_M = st.text_input('Upper X line (monthly)', value=str(horizontal_max_m))
            try:
                horizontal_Upper_M = float(horizontal_Upper_M)
            except ValueError:
                horizontal_Upper_M = horizontal_max_m
                horizontal_Upper_M = round(horizontal_Upper_M, 2)

            # รับค่าเส้นแนวตั้งล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            vertical_Lower_M = st.text_input('Lower Y line (monthly)', value=str(vertical_min_m))
            try:
                vertical_Lower_M = float(vertical_Lower_M)
            except ValueError:
                vertical_Lower_M = vertical_min_m
                vertical_Lower_M = round(vertical_Lower_M, 2)

            # รับค่าเส้นแนวตั้งบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            vertical_Upper_M = st.text_input('Upper Y line (monthly)', value=str(vertical_max_m))
            try:
                vertical_Upper_M = float(vertical_Upper_M)
            except ValueError:
                vertical_Upper_M = vertical_max_m
                vertical_Upper_M = round(vertical_Upper_M, 2)

            # ตรวจสอบว่าค่าที่ใส่มาอยู่ในช่วงที่ถูกต้องหรือไม่
            if horizontal_Lower_M >= horizontal_Upper_M or vertical_Lower_M >= vertical_Upper_M:
                st.error("ค่า Lower ต้องน้อยกว่าค่า Upper สำหรับข้อมูลรายเดือน.")
                st.stop()

                        
            data_classified = classify_points_monthly(data_1, vertical_Lower_M, vertical_Upper_M, horizontal_Lower_M, horizontal_Upper_M)

        with input_col2:
                n_clusters_monthly = st.slider('Select number of clusters for monthly', 2, 10, 3)
                clustered_data_monthly = kmeans_clustering_monthly(data_classified, n_clusters_monthly)
                # ใช้ค่าที่ถูกต้องจาก sliders สำหรับเดือน
                scatter_plot_monthly(data_m=clustered_data_monthly, title='Clusters Monthly',
                                    vertical_lower_m=vertical_Lower_M,
                                    vertical_upper_m=vertical_Upper_M,
                                    horizontal_lower_m=horizontal_Lower_M,
                                    horizontal_upper_m=horizontal_Upper_M)
                


        st.markdown('<h4 style="text-align:center;">🔍Clustered Data - Monthly</h4>', unsafe_allow_html=True)
        st.markdown('')
        st.dataframe(clustered_data_monthly[['Product','Grade','Gram','average_monthly', 'std_ton_monthly', 'Cluster', 'Area']],width=1500, height=300)
        st.markdown('')
        st.markdown('')
        st.markdown('<h4 style="text-align:center;">🔢Count Tonnage and Product From 9-Box</h4>', unsafe_allow_html=True)
        st.markdown("""
                        <style>
                        .metric-box {
                            border: 1px solid #e1e4e8; /*สีขอบกรอบ*/
                            background-color: #C0DEFF; /*เปลี่ยนสีพื้นหลังกรอบ*/
                            border-radius: 5px; 
                            padding: 5px 10px;
                            margin: 10px 0;
                            text-align: center;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        }
                        .metric-box h2 {
                            font-size: 1rem;
                            color: #888; /* เปลี่ยนสีข้อความหัว */
                            margin: 0;
                        }
                        .metric-box h1 {
                            font-size: 2.5rem;
                            color: #000;
                            margin: 0;
                        }
                        </style>
                    """, unsafe_allow_html=True)

        first_row_cols = st.columns(3)
        second_row_cols = st.columns(3) 
        cluster_type={
            0:'non-std', 1:'low-std',2:'std'
        }
        area_type = {
            1: 'std', 2: 'std', 3: 'low-std', 
            4: 'std', 5: 'low-std', 6: 'non-std', 
            7: 'low-std', 8: 'non-std', 9: 'non-std'
        }
        
        data_1['Product Type'] = data_1['Area'].map(area_type)
        data_1['Product Type Cluster'] = data_1['Cluster'].map(cluster_type)
        
        
        grouped_data = data_1.groupby('Product Type').agg(
            Total_Ton=('ton', 'sum'),  
            Product_Count=('Product', 'count') 
        ).reset_index()
        
        # grouped_data
        first_row_cols = st.columns(3)
        for i, category in enumerate(['std', 'low-std', 'non-std']):
            if category in grouped_data['Product Type'].values:
                total_ton = grouped_data.loc[grouped_data['Product Type'] == category, 'Total_Ton'].values[0]
                formatted_total_ton = "{:,.3f}".format(total_ton)
            else:
                formatted_total_ton = "0.000"
            with first_row_cols[i]:
                st.markdown(f"""
                    <div class="metric-box">
                    <h2>Total Ton ({category})</h2>
                    <h1>{formatted_total_ton}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
        second_row_cols = st.columns(3)
        for i, category in enumerate(['std', 'low-std', 'non-std']):
            if category in grouped_data['Product Type'].values:
                product_count = grouped_data.loc[grouped_data['Product Type']== category, 'Product_Count'].values[0]
            else:
                product_count  ="0"
            with second_row_cols[i]:
                st.markdown(f"""
                    <div class="metric-box">
                    <h2>Product Count ({category})</h2>
                    <h1>{product_count}</h1>
                    </div>
                    
                    """, unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('<h4 style="text-align:center;">🧮Dataframe</h4>', unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.dataframe(data_1[['Product','Grade','Gram','ton','number_of_month','average_monthly','std_ton_monthly','Product Type','Product Type Cluster' ]],width=1500, height=300)
        st.stop()           

            
    # ✋🏻 ---------------------------------------------------------------------------------------------------------------------------        
    
    elif selected == "9-Box":
        input_col1,input_col2 = st.columns([1,2])
        with input_col1:
            st.markdown('')
            st.markdown('')
            st.markdown('')
            st.markdown('##### Input Number In TextBox⬇️')
            # ใช้ข้อมูลที่มีอยู่เพื่อกำหนดค่าเริ่มต้นให้กับช่องใส่ข้อมูล
            horizontal_min_m = round(float(data_1['std_ton_weekly'].min()), 2)
            horizontal_max_m = round(float(data_1['std_ton_weekly'].max()), 2)
            vertical_min_w = round(float(data_1['average_monthly'].min()), 2)
            vertical_max_w = round(float(data_1['average_monthly'].max()), 2)

            # รับค่าเส้นแนวนอนล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            horizontal_Lower_M = st.text_input('avg-monthly-lower(x)', value=str(horizontal_min_m))
            try:
                horizontal_Lower_M = float(horizontal_Lower_M)
            except ValueError:
                horizontal_Lower_M = horizontal_min_m
                horizontal_Lower_M = round(horizontal_Lower_M, 2)

            # รับค่าเส้นแนวนอนบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            horizontal_Upper_M = st.text_input('avg-monthly-upper(x)', value=str(horizontal_max_m))
            try:
                horizontal_Upper_M = float(horizontal_Upper_M)
            except ValueError:
                horizontal_Upper_M = horizontal_max_m
                horizontal_Upper_M = round(horizontal_Upper_M, 2)

            # รับค่าเส้นแนวตั้งล่างจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            vertical_Lower_W = st.text_input('cv-lower(y)', value=str(vertical_min_w))
            try:
                vertical_Lower_W = float(vertical_Lower_W)
            except ValueError:
                vertical_Lower_W = vertical_min_w
                vertical_Lower_W = round(vertical_Lower_W, 2)

            # รับค่าเส้นแนวตั้งบนจากช่องใส่ข้อความ-แปลงเป็นจำนวนทศนิยม
            vertical_Upper_W = st.text_input('cv-upper(y)', value=str(vertical_max_w))
            try:
                vertical_Upper_W = float(vertical_Upper_W)
            except ValueError:
                vertical_Upper_W = vertical_max_w
                vertical_Upper_W = round(vertical_Upper_W, 2)

            # ตรวจสอบว่าค่าที่ใส่มาอยู่ในช่วงที่ถูกต้องหรือไม่
            if horizontal_Lower_M >= horizontal_Upper_M or vertical_Lower_W >= vertical_Upper_W:
                st.error("ค่า Lower ต้องน้อยกว่าค่า Upper.")
                st.stop()

                        
            data_classified = classify_points_9box(data_1, vertical_Lower_W, vertical_Upper_W, horizontal_Lower_M, horizontal_Upper_M)

        with input_col2:
                n_clusters_9box = st.slider('Select number of clusters for 9-box', 2, 10, 4)
                clustered_data_9box = kmeans_clustering_9box(data_classified, n_clusters_9box)
                scatter_plot_9box(data_9box=clustered_data_9box, title='Clusters 9-Box', 
                                    vertical_lower_w=vertical_Lower_W, 
                                    vertical_upper_w=vertical_Upper_W, 
                                    horizontal_lower_m=horizontal_Lower_M, 
                                    horizontal_upper_m=horizontal_Upper_M)
                


        # st.markdown('#### 🔍Clustered Data 9-Box')
        st.markdown('<h4 style="text-align:center;">🔍Clustered Data 9-Box</h4>', unsafe_allow_html=True)
        st.markdown('')
        st.dataframe(clustered_data_9box[['Product','Grade','Gram','average_monthly', 'std_ton_weekly', 'Cluster', 'Area']],width=1500, height=300)
        st.markdown('')
        st.markdown('')
        st.markdown('<h4 style="text-align:center;">🔢Count Tonnage and Product From 9-Box</h4>', unsafe_allow_html=True)
        
        st.markdown("""
                        <style>
                        .metric-box {
                            border: 1px solid #e1e4e8; /*สีขอบกรอบ*/
                            background-color: #D4D2F2; /*เปลี่ยนสีพื้นหลังกรอบ*/
                            border-radius: 5px; 
                            padding: 5px 10px;
                            margin: 10px 0;
                            text-align: center;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        }
                        .metric-box h2 {
                            font-size: 1rem;
                            color: #888; /* เปลี่ยนสีข้อความหัว */
                            margin: 0;
                        }
                        .metric-box h1 {
                            font-size: 2.5rem;
                            color: #000;
                            margin: 0;
                        }
                        </style>
                    """, unsafe_allow_html=True)

        first_row_cols = st.columns(3)
        second_row_cols = st.columns(3) 
        cluster_type={
            0:'non-std', 1:'low-std',2:'std'
        }
        area_type = {
            1: 'std', 2: 'std', 3: 'low-std', 
            4: 'std', 5: 'low-std', 6: 'non-std', 
            7: 'low-std', 8: 'non-std', 9: 'non-std'
        }
        data_1['Product Type'] = data_1['Area'].map(area_type)
        data_1['Product Type Cluster'] = data_1['Cluster'].map(cluster_type)

        #คำนวนผลรวม ตัน / นับ product 
        grouped_data = data_1.groupby('Product Type').agg(
            Total_Ton=('ton', 'sum'),  
            Product_Count=('Product', 'count') 
        ).reset_index()

        first_row_cols = st.columns(3)
        for i, category in enumerate(['std', 'low-std', 'non-std']):
            total_ton = grouped_data.loc[grouped_data['Product Type'] == category, 'Total_Ton'].values[0]
            formatted_total_ton = "{:,.3f}".format(total_ton)
            with first_row_cols[i]:
                st.markdown(f"""
                    <div class="metric-box">
                    <h2>Total Ton ({category})</h2>
                    <h1>{formatted_total_ton}</h1>
                    </div>
                    """, unsafe_allow_html=True)

        # สร้างแถวที่สองสำหรับแสดงข้อมูล Product Count
        second_row_cols = st.columns(3)
        for i, category in enumerate(['std', 'low-std', 'non-std']):
            product_count = grouped_data.loc[grouped_data['Product Type'] == category, 'Product_Count'].values[0]
            with second_row_cols[i]:
                st.markdown(f"""
                    <div class="metric-box">
                    <h2>Product Count ({category})</h2>
                    <h1>{product_count}</h1>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('')
        st.markdown('')
        st.markdown('<h4 style="text-align:center;">🧮Dataframe</h4>', unsafe_allow_html=True)
        st.markdown('')
        # st.dataframe(data_1[['Product','Grade','Gram','ton','number_of_week','number_of_month','average_weekly','average_monthly','std_ton_weekly','std_ton_monthly','Product Type', ]],width=1500, height=400)
        st.dataframe(data_1[['Product','Grade','Gram','ton','number_of_week','number_of_month','average_monthly','std_ton_weekly','Product Type','Product Type Cluster' ]],width=1500, height=400)
        st.stop()
    # ✋🏻 ---------------------------------------------------------------------------------------------------------------------------        

    if selected == "Calculate":
        st.title('Calculator :blue[Inventory Management] 🧮')
        uploaded_file_calculate = st.file_uploader('Upload Excel File for Calculation', type=['xlsx', 'csv'], key='calculate_uploader')
    if uploaded_file_calculate is not None:
        st.session_state['uploaded_file_calculate'] = uploaded_file_calculate
        data_2 = load_data(uploaded_file_calculate)
        # st.write(data_2)

    col1,col2 ,col3 = st.columns(3)
    with col1:
        percentile_input_high = st.number_input('Enter the percentile for high (0-100):', min_value=0.0, max_value=100.0, value=97.0)
        z_score_high = stats.norm.ppf(percentile_input_high / 100.0)
        # z_score_high = stats.norm.ppf(percentile_input_high) 
        st.write(f"Z-Score High: {z_score_high:.4f}")
    with col2:
        percentile_input_middle = st.number_input('Enter the percentile for middle (0-100):', min_value=0.0, max_value=100.0, value=93.0)
        z_score_middle = stats.norm.ppf(percentile_input_middle / 100.0)
        # z_score_middle = stats.norm.ppf(percentile_input_middle)
        st.write(f"Z-Score Middle: {z_score_middle:.4f}")
    with col3:
        percentile_input_low = st.number_input('Enter the percentile for low (0-100):', min_value=0.0, max_value=100.0, value=65.0)
        z_score_low = stats.norm.ppf(percentile_input_low / 100.0)
        # z_score_low = stats.norm.ppf(percentile_input_low)
        st.write(f"Z-Score Low: {z_score_low:.4f}")

    st.markdown('')
    st.markdown('')
    def calculate_z_score(row):
        if row['Product Type'] == 'std':
            return z_score_high
        elif row['Product Type'] == 'low-std':
            return z_score_middle
        elif row['Product Type'] == 'non-std':
            return z_score_low
        else:
            return None  
    
    def calculate_z_score_cluster(row):
        if row['Product Type Cluster'] == 'std':
            return z_score_high
        elif row['Product Type Cluster'] == 'low-std':
            return z_score_middle
        elif row['Product Type Cluster'] == 'non-std':
            return z_score_low
        else:
            return None  
    
    data_2['Z_score'] = data_2.apply(calculate_z_score, axis=1)
    data_2['Z_std'] = data_2['Z_score'] *(data_2['std_ton_weekly'])
    data_2['Z_score_cluster'] = data_2.apply(calculate_z_score_cluster, axis=1)
    data_2['Z_std_cluster'] = data_2['Z_score_cluster'] *(data_2['std_ton_weekly'])
    
    # st.write(data_2)
    np.random.seed(42)  
    data_2['Lead Time (weeks)'] = np.random.randint(1, 12, data_2.shape[0])
    data_2['std_leadtime'] = data_2.groupby('Product')['Lead Time (weeks)'].transform('std')
    data_2['avg_leadtime'] = data_2.groupby('Product')['Lead Time (weeks)'].transform('mean')
    
    data_2['Safety Stock'] = data_2['Z_std'] * np.sqrt(data_2['avg_leadtime']) 
    data_2['ROP Weekly'] = ((data_2['average_weekly']*data_2['avg_leadtime']) + data_2['Safety Stock'])
    
    data_2['Safety Stock Cluster'] = data_2['Z_std_cluster'] * np.sqrt(data_2['avg_leadtime']) 
    data_2['ROP Weekly Cluster'] = ((data_2['average_weekly']*data_2['avg_leadtime']) + data_2['Safety Stock Cluster'])
   
    data_2 = data_2.drop(columns=['Unnamed: 0'])
    
    tab1 , tab2 = st.tabs(["Safety Stock & Reorder Point ⚠️", "Safety Stock & Reorder Point (Cluster)⚠️"])
    with tab1 :
        st.markdown('<h4 style="text-align:center;">🛡️ Safety Stock & Reorder Point ⚠️</h4>', unsafe_allow_html=True)
        st.dataframe(data_2[['Product','Grade','Gram','ton','number_of_week','average_weekly','std_ton_weekly','Product Type','Z_score'
                            ,'Z_std','Lead Time (weeks)','std_leadtime','avg_leadtime','Safety Stock','ROP Weekly' ]],width=1500, height=400)  
    with tab2 :    
        st.markdown('<h4 style="text-align:center;color:red;">🛡️ Safety Stock & Reorder Point (Cluster)⚠️</h4>', unsafe_allow_html=True)
        st.dataframe(data_2[['Product','Grade','Gram','ton','number_of_week','average_weekly','std_ton_weekly','Product Type Cluster','Z_score_cluster'
                            ,'Z_std_cluster','Lead Time (weeks)','std_leadtime','avg_leadtime','Safety Stock Cluster','ROP Weekly Cluster' ]],width=1500, height=400)
    
    product_list = data_2.apply(lambda x: f"{x['Product']} - {x['Grade']} - {x['Gram']}g", axis=1).unique().tolist()

 
    selected_product = st.selectbox('Select a product (Product : Grade : Gram)', product_list)

    selected_parts = selected_product.split(" - ")
    selected_product_data = data_2[
        (data_2['Product'] == selected_parts[0]) & 
        (data_2['Grade'] == selected_parts[1]) & 
        (data_2['Gram'] == int(selected_parts[2][:-1])) 
    ]

    # แสดงข้อมูลของผลิตภัณฑ์ที่เลือก
    st.dataframe(selected_product_data)   
    st.markdown("""
                    <style>
                    .metric-box {
                        border: 4px solid #15F5BA; /*สีขอบกรอบ*/
                        /*background-color: #F8FFDB;  เปลี่ยนสีพื้นหลังกรอบ*/
                        border-radius: 5px; 
                        padding: 5px 10px;
                        margin: 10px 0;
                        text-align: center;
                        box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    }
                    .metric-box h2 {
                        font-size: 1.2rem;
                        color: #000000; /* เปลี่ยนสีข้อความหัว */
                        margin: 0;
                    }
                    .metric-box h1 {
                        font-size: 2.5rem;
                        color: #836FFF;
                        margin: 0;
                    }
                    </style>
                """, unsafe_allow_html=True)
    result = st.columns(3)
    with result[0]:
        st.markdown(f'<div class="metric-box"><h2>Safety Stock</h2><h1>{selected_product_data["Safety Stock"].sum():.4f}</h1></div>', unsafe_allow_html=True)

    with result[1]:
                st.markdown(f'<div class="metric-box"><h2>Reoder Point</h2><h1>{selected_product_data["ROP Weekly"].sum():.4f}</h1></div>', unsafe_allow_html=True)

    with result[2]:
        st.markdown('<div class="metric-box"><h2>Shotage Cost</h2><h1>300</h1></div>', unsafe_allow_html=True)
        
    
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    options = data_2.apply(lambda x: f"{x['Product']} - {x['Grade']} - {x['Gram']}", axis=1).unique().tolist()
    selected_products = st.multiselect('Select a product (Product : Grade : Gram)', options)
    selected_product_data = pd.DataFrame()

    for product in selected_products:
        selected_parts = product.split(" - ")
        mask = (data_2['Product'] == selected_parts[0]) & \
            (data_2['Grade'] == selected_parts[1]) & \
            (data_2['Gram'] == int(selected_parts[2]))  
        selected_product_data = pd.concat([selected_product_data, data_2[mask]])

    st.dataframe(selected_product_data)
    
    

    # st.dataframe(data_2)
    
    
if __name__ == "__main__":
    main()


