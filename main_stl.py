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
from weekly_data import WeeklyData
from monthly_data import MonthlyData
from box import Box  


@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def main():
    st.set_page_config(page_title='Dashboard for data analysis', page_icon='📉', layout="wide")
    st.sidebar.image("data.png", caption="Dashboard for data analysis")

    with st.sidebar:
        selected = option_menu("Main Menu", ["Weekly Data", "Monthly Data","9-Box"], 
                               icons=["calendar-week", "calendar-month","box"], default_index=0)

    st.title('Cluster :green[Analysis] with Streamlit📊📉')
    uploaded_file = st.file_uploader('Upload Excel File (Dataset 1)', type=['xlsx'])
    

    if uploaded_file is not None:
        data_1 = load_data(uploaded_file)
        weekly = WeeklyData(data_1)
        
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
                data_classified = weekly.classify_points_weekly(vertical_Lower_W, vertical_Upper_W, horizontal_Lower_W, horizontal_Upper_W)
            
            with input_col2:
                    n_clusters_weekly = st.slider('Select number of clusters for weekly', 2, 10, 4)
                    clustered_data_weekly = weekly.kmeans_clustering_weekly(n_clusters_weekly)
                    weekly.scatter_plot_weekly('Clusters Weekly', vertical_Lower_W, vertical_Upper_W, horizontal_Lower_W, horizontal_Upper_W )
                    # area_counts_weekly = clustered_data_weekly['Area'].value_counts()
        
            

            st.markdown('#### 🔍Clustered Data - Weekly')
            st.dataframe(clustered_data_weekly[['Product','Grade','Gram','average_weekly', 'std_ton_weekly', 'Cluster', 'Area']],width=1500, height=300)
            
            st.markdown('#### 🔢Count The Area Of ​The Week')
            area_counts_weekly = clustered_data_weekly['Area'].value_counts().sort_index()

            st.markdown("""
                            <style>
                            .metric-box {
                                border: 1px solid #e1e4e8; /*สีขอบกรอบ*/
                                background-color: #F6C8B6; /*เปลี่ยนสีพื้นหลังกรอบ*/
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

            # แสดงข้อมูลในแถวแรก (area 1-5)
            first_row_cols = st.columns(5)
            for i in range(1, 6):  # Area 1-5
                count = area_counts_weekly.get(i, 0)
                with first_row_cols[i-1]: # index for columns starts at 0
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i}</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
            second_row_cols = st.columns(5)
            for i in range(6, 10): # Area 6-9
                count = area_counts_weekly.get(i, 0)
            # Fill the first four columns with Area 6-9, leave the fifth column empty
                with second_row_cols[i-6] if i < 10 else st.empty():
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i}</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                with second_row_cols[4]:
                    st.markdown("""<div class="metric-box"></div>""", unsafe_allow_html=True)
                    
            st.markdown('')
            st.markdown('')
            st.markdown('<h4 style="text-align:center;">🧮Calculate The Tonnage In Each Area</h4>', unsafe_allow_html=True)

            input_col3,input_col4 = st.columns([1,2]) 
            with input_col3:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                
                # cal ton
                area_sums_weekly = clustered_data_weekly.groupby('Area')['ton'].sum().reset_index()
                result_df = pd.DataFrame({
                    'Area': area_sums_weekly['Area'],
                    'Total Ton': area_sums_weekly['ton']
                })

                st.dataframe(result_df) 
            with input_col4:    
                colors = ['#FFA07A', '#20B2AA', '#778899', '#B0C4DE', '#FFFFE0', '#00FA9A', '#FFD700', '#87CEFA', '#FF69B4']   
                fig = go.Figure(go.Bar(  
                    x=result_df['Area'], 
                    y=result_df['Total Ton'],
                    text=result_df['Total Ton'],  
                    marker_color=colors,  
                    textposition='outside' 
                    ))

                fig.update_traces(
                    hoverinfo='name+y',  
                    textfont_size=20,
                    marker=dict(line=dict(color='white', width=2))
                )

                # Update layout for a better view
                fig.update_layout(
                    xaxis=dict(title='Area'),
                    yaxis=dict(title='Total Ton'),
                    barmode='group',
                    uniformtext_minsize=8,  # Ensure the text size
                    uniformtext_mode='hide'  # Hide text if bar is too small
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig)
                
    # ✋🏻 ---------------------------------------------------------------------------------------------------------------------------        
        
        elif selected == "Monthly Data":
            monthly = MonthlyData(data_1)
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

                            
                data_classified = monthly.classify_points_monthly(vertical_Lower_M, vertical_Upper_M, horizontal_Lower_M, horizontal_Upper_M)

            with input_col2:
                    n_clusters_monthly = st.slider('Select number of clusters for monthly', 2, 10, 4)
                    clustered_data_monthly = monthly.kmeans_clustering_monthly(n_clusters_monthly)
                    # ใช้ค่าที่ถูกต้องจาก sliders สำหรับเดือน
                    monthly.scatter_plot_monthly('Clusters Monthly', vertical_Lower_M, vertical_Upper_M, horizontal_Lower_M, horizontal_Upper_M )    
                    


            st.markdown('#### 🔍Clustered Data - Monthly')
            st.dataframe(clustered_data_monthly[['Product','Grade','Gram','average_monthly', 'std_ton_monthly', 'Cluster', 'Area']],width=1500, height=300)

            st.markdown('#### 🔢Count The Area Of ​​The Month')
            area_counts_monthly = clustered_data_monthly['Area'].value_counts().sort_index()

            st.markdown("""
                            <style>
                            .metric-box {
                                border: 1px solid #e1e4e8; /*สีขอบกรอบ*/
                                background-color: #A8D1E7; /*เปลี่ยนสีพื้นหลังกรอบ*/
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

            # แสดงข้อมูลในแถวแรก (areas 1-5)
            first_row_cols = st.columns(5)
            for i in range(1, 6):  # Area 1-5
                count = area_counts_monthly.get(i, 0)
                with first_row_cols[i-1]: # index for columns starts at 0
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i}</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
            second_row_cols = st.columns(5)
            for i in range(6, 10): # Area 6-9
                count = area_counts_monthly.get(i, 0)
            # Fill the first four columns with Area 6-9, leave the fifth column empty
                with second_row_cols[i-6] if i < 10 else st.empty():
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i}</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                with second_row_cols[4]:
                    st.markdown("""<div class="metric-box"></div>""", unsafe_allow_html=True)
            
            st.markdown('')
            st.markdown('')
            st.markdown('<h4 style="text-align:center;">🧮Calculate The Tonnage In Each Area</h4>', unsafe_allow_html=True)

            input_col3,input_col4 = st.columns([1,2]) 
            with input_col3:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                
                # cal ton
                area_sums_monthly = clustered_data_monthly.groupby('Area')['ton'].sum().reset_index()
                result_df = pd.DataFrame({
                    'Area': area_sums_monthly['Area'],
                    'Total Ton': area_sums_monthly['ton']
                })

                st.dataframe(result_df) 
            with input_col4:    
                colors = ['#FFA07A', '#20B2AA', '#778899', '#B0C4DE', '#FFFFE0', '#00FA9A', '#FFD700', '#87CEFA', '#FF69B4']   
                fig = go.Figure(go.Bar(  
                    x=result_df['Area'], 
                    y=result_df['Total Ton'],
                    text=result_df['Total Ton'],  
                    marker_color=colors,  
                    textposition='outside' 
                    ))

                fig.update_traces(
                    hoverinfo='name+y',  
                    textfont_size=20,
                    marker=dict(line=dict(color='white', width=2))
                )

                # Update layout for a better view
                fig.update_layout(
                    xaxis=dict(title='Area'),
                    yaxis=dict(title='Total Ton'),
                    barmode='group',
                    uniformtext_minsize=8,  # Ensure the text size
                    uniformtext_mode='hide'  # Hide text if bar is too small
                )
                st.plotly_chart(fig)
                
        # ✋🏻 ---------------------------------------------------------------------------------------------------------------------------        
        
        elif selected == "9-Box":
            box = Box(data_1)
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

                            
                data_classified = box.classify_points_9box( vertical_Lower_W, vertical_Upper_W, horizontal_Lower_M, horizontal_Upper_M)

            with input_col2:
                    n_clusters_9box = st.slider('Select number of clusters for 9-box', 2, 10, 4)
                    clustered_data_9box = box.kmeans_clustering_9box(n_clusters_9box)
                    box.scatter_plot_9box('Clusters 9-Box',vertical_Lower_W,vertical_Upper_W,horizontal_Lower_M,horizontal_Upper_M)
                    
            st.markdown('#### 🔍Clustered Data 9-Box')
            st.dataframe(clustered_data_9box[['Product','Grade','Gram','average_monthly', 'std_ton_weekly', 'Cluster', 'Area']],width=1500, height=300)

            st.markdown('#### 🔢Count The Area From 9-Box')
            area_counts_9box = clustered_data_9box['Area'].value_counts().sort_index()

            
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

            first_row_cols = st.columns(5)
            second_row_cols = st.columns(5) 
            area_type = {
                1: 'std', 2: 'std', 3: 'low-std', 
                4: 'std', 5: 'low-std', 6: 'non-std', 
                7: 'low-std', 8: 'non-std', 9: 'non-std'
            }
           
            first_row_cols = st.columns(5)
            for i in range(1, 6):  
                count = area_counts_9box.get(i, 0)
                with first_row_cols[i-1]: 
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i} ({area_type[i]})</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
            second_row_cols = st.columns(5)
            for i in range(6, 10): 
                count = area_counts_9box.get(i, 0)
                with second_row_cols[i-6] if i < 10 else st.empty():
                    st.markdown(f"""
                        <div class="metric-box">
                        <h2>Area {i} ({area_type[i]})</h2>
                        <h1>{count}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                with second_row_cols[4]:
                    st.markdown("""<div class="metric-box"></div>""", unsafe_allow_html=True)



            st.markdown('')
            st.markdown('')
            st.markdown('<h4 style="text-align:center;">🧮Calculate The Tonnage In Each Area</h4>', unsafe_allow_html=True)

            input_col3,input_col4 = st.columns([1,2]) 
            with input_col3:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                
                # Product Type 3 type ที่เอามารวมกัน
                clustered_data_9box['Product Type'] = clustered_data_9box['Area'].map(area_type)
                ton_sum_by_area = clustered_data_9box.groupby('Product Type')['ton'].sum().reset_index()
                sorted_data = clustered_data_9box.sort_values(by=['Product Type', 'ton'], ascending=[True, False])
            with input_col4:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                # area_sums_monthly = clustered_data_9box.groupby('Area')['ton'].sum().reset_index()
                # result_df = pd.DataFrame({
                #     'Area': area_sums_monthly['Area'],
                #     'Total Ton': area_sums_monthly['ton']
                # })
                # st.dataframe(result_df) 
                # st.dataframe(ton_sum_by_area[['Product Type','ton']])
            st.dataframe(sorted_data[['Product','Grade','Gram','ton','number_of_week','number_of_month','average_weekly','average_monthly','std_ton_weekly','std_ton_monthly','Product Type', ]],width=1500, height=300)
                
            result = pd.DataFrame({
                    'Product Type' : ton_sum_by_area['Product Type'],
                    'Total Ton': ton_sum_by_area['ton']
                })
            st.dataframe(result)
           

                
                
            
if __name__ == "__main__":
    main()