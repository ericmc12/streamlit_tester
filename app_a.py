import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import json
from streamlit_lottie import st_lottie

import pathlib
#plt = platform.system()
#if plt == 'Windows': pathlib.WindowsPath = pathlib.PosixPat
from pathlib import Path




#####################################
# Streamlit page configuration
#####################################

#st.set_page_config(
#    page_title="Sign Language Classifier",
#     page_icon="🤟",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'How to train model': 'https://colab.research.google.com/drive/1ZfCMJ55adGJqu2tbIxfFIUHsl0rEdaWn?usp=sharing',
#         'Report a bug': "https://github.com/ericmc12/streamlit_tester/issues",
#         'About': "# This is a capstone project."
#     }
# )


st.set_page_config(
     page_title="Sign Language Classifier",
     page_icon="🤟",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://colab.research.google.com/drive/1ZfCMJ55adGJqu2tbIxfFIUHsl0rEdaWn?usp=sharing',
         'Report a bug': "https://github.com/ericmc12/streamlit_tester/issues",
         'About': "# This is a capstone project."
     }
 )







#####################################
# Display lottie file 
#####################################

st.markdown("<h1 style='text-align: center;'>Welcome, I am your personal Sign Language Translator.</h1>", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
        
lottie_coding = load_lottiefile("images/lottie.json")
col1, col2, col3 = st.columns(3)
with col2:   
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="medium", # medium ; high,
        key=None,
        )
    


#####################################
# Load model 
#####################################

#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

def load_model():
    return load_learner("new2.pkl")

with st.spinner("Loading...."):
    model = load_model()
    
#####################################
# Upload image and Classify 
#####################################

uploaded_image = st.file_uploader("Upload your image and I'll give it a try.", type=["png", "jpg"])
if uploaded_image is not None:
    
    st.image(uploaded_image)
    pred,pred_idx,probs = model.predict(uploaded_image.getvalue())
    st.success(f"The letter: {pred} ; Probability: {probs[pred_idx]:.04f}")
    st.caption(f"Caution: I have only been trained on a small set of images. I may also be wrong.")
   

#####################################
# Infos
#####################################
    
with st.expander("Info"):
     st.markdown("""
         - I have been trained by fine-tuning a __ResNet18__ convolutional neural network
         - For each letter, I have been provided around 500 images to learn from
         - After 3 training runs (epochs), this was the result on the validation set:    
     """)
     st.image("images/confusionmatrix.png")
     st.markdown("""
         Want to view the code?
         [Google Colab](https://colab.research.google.com/drive/1ZfCMJ55adGJqu2tbIxfFIUHsl0rEdaWn?usp=sharing)   
     """)
     
if st.button("Press button to load example image"):
    example_image = "images/example_image.jpg" 
    st.image(example_image)
    pred,pred_idx,probs = model.predict(example_image)
   
    st.success(f"The letter: {pred} ; Probability: {probs[pred_idx]:.04f}")
