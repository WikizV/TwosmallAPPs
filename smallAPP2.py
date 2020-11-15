#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

from PIL import Image,ImageFilter,ImageEnhance

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and Subheader
st.title("Small Business EDA App")
st.subheader("EDA Web App with Streamlit ")

# EDA
my_dataset = "final_processed_data_0.csv"

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
    df = pd.read_csv('G://google download/Research/UBS_pitch/final_processed_data_0.csv')
    #df2 = pd.read_csv('G://google download/Research/UBS_pitch/closure_history_1.csv')
    return df 

def explore_data_2(dataset):
    df_2 = pd.read_csv('G://google download/Research/UBS_pitch/closure_history_1.csv')
    #df2 = pd.read_csv('G://google download/Research/UBS_pitch/closure_history_1.csv')
    return df_2 

# Show Dataset
if st.checkbox("Preview DataFrame"):
    data = explore_data(my_dataset)
    if st.button("Head"):
        st.write(data.head())
    if st.button("Tail"):
        st.write(data.tail())
    else:
        st.write(data.head(2))
        
# Show Entire Dataframe
if st.checkbox("Show All DataFrame"):
    data = explore_data(my_dataset)
    st.dataframe(data)
    
# Show Description
if st.checkbox("Show All Column Name"):
    data = explore_data(my_dataset)
    st.text("Columns:")
    st.write(data.columns)
    
# Dimensions
data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
if data_dim == 'Rows':
    data = explore_data(my_dataset)
    st.text("Showing Length of Rows")
    st.write(len(data))
if data_dim == 'Columns':
    data = explore_data(my_dataset)
    st.text("Showing Length of Columns")
    st.write(data.shape[1])

# Summary of the Dataframe
if st.checkbox("Show Summary of Dataset"):
    data = explore_data(my_dataset)
    st.write(data.describe())
    
# Selection
species_option = st.selectbox('Select Columns',('Company Name','Address','City','ZIP Code',
                                                'Metro Area',
                                                'Sales_Change',
                                                'Grocery_within_Zip','County','Civilian_labor_force_2019',
                                                'Median_Household_Income_2018',
                                               'Location Employee Size Actual',
                                               'Location Sales Volume Actual','Years In Database',
                                                'Credit Score Alpha','Supermarket_within_5_miles',
                                                'Square Footage','Google_Scores','Google_Reviews'))
data = explore_data(my_dataset)

    
if species_option == 'Company Name':
    st.write(data['Company Name'])
    
elif species_option == 'Address':
    st.write(data['Address'])
    
elif species_option == 'City':
    st.write(data['City'])
    
elif species_option == 'ZIP Code':
    st.write(data['ZIP Code'])
    
elif species_option == 'Metro Area':
    st.write(data['Metro Area'])

elif species_option == 'Sales_Change':
    st.write(data['Sales_Change'])

elif species_option == 'Google_Reviews':
    st.write(data['Google_Reviews'])
    
elif species_option == 'Google_Scores':
    st.write(data['Google_Scores'])
    
elif species_option == 'Grocery_within_Zip':
    st.write(data['Grocery_within_Zip'])
    
elif species_option == 'Civilian_labor_force_2019':
    st.write(data['Civilian_labor_force_2019'])
    
elif species_option == 'Median_Household_Income_2018':
    st.write(data['Median_Household_Income_2018'])    
elif species_option == 'Location Employee Size Actual':
    st.write(data['Location Employee Size Actual']) 
elif species_option == 'County':
    st.write(data['County']) 
elif species_option == 'Years In Database':
    st.write(data['Years In Database']) 
elif species_option == 'Location Sales Volume Actual':
    st.write(data['Location Sales Volume Actual']) 
elif species_option == 'Square Footage':
    st.write(data['Square Footage'])
elif species_option == 'Supermarket_within_5_miles':
    st.write(data['Supermarket_within_5_miles'])
elif species_option == 'Credit Score Alpha':
    st.write(data['Credit Score Alpha'])
else:
    st.write("Select A Column")
    
# Show Plots
# if st.checkbox("Simple Bar Plot with Matplotlib"):
#     data = explore_data(my_dataset)
#     data.plot(kind='bar')
#     st.pyplot()
    
# Show Plots
if st.checkbox("Simple Correlation Plot with Seaborn "):
    data = explore_data(my_dataset)
    st.write(sns.heatmap(data.corr(),vmax=0.9,square=True))
    # Use Matplotlib to render seaborn
    st.pyplot()
    
    
if st.checkbox("Closure history of grocery stores in New Jersey"):
    data_2 = explore_data_2(my_dataset)
    #st.write(
    fig2 = px.scatter_mapbox(data_2, lat="lat", lon="long", zoom=7, height=700,hover_name='Company Name',
                        hover_data=['Status','Location Sales Volume Actual',
                                    'Location Employee Size Actual','Supermarket_within_5_miles'],
                        #color_continuous_scale='Plasma',,
                        color='Status', color_discrete_sequence=["green", "red"],
                         #size='PointSize',
                         animation_frame='Year'
                        )
    #fig2.update_layout(title_text = 'History of closed grocery stores in New Jersey',font_size=18)
    fig2.update_layout(mapbox_style="open-street-map")
    fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig2.update_layout(margin=dict(l=20,r=0,b=0,t=70,pad=0),paper_bgcolor="white",font_size=18)
    st.plotly_chart(fig2)

       # )
    # Use Matplotlib to render seaborn
    #st.pyplot()



# Show Plots
# if st.checkbox("Bar Plot of Groups or Counts"):
#     data = explore_data(my_dataset)
#     v_counts = data.groupby('species')
#     st.bar_chart(v_counts)

# About

if st.button("About App"):
    st.subheader("SmallBusiness Dataset EDA App")
    st.text("Built with LMWGY Team")
    st.text("Thanks for Watching")

if st.checkbox("By"):
    st.text("LMWGY Team")


# In[ ]:




