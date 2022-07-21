import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
import json
from streamlit_lottie import st_lottie

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPat
from pathlib import Path




#####################################
# Streamlit page configuration
#####################################

st.set_page_config(
     page_title="Sign Language Classifier",
     page_icon="🍌",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://github.com/IBronko/',
         'Report a bug': "https://github.com/IBronko/fruit-image-classifier/issues",
         'About': "# This is a personal project."
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
# Modify model answers
#####################################

fruit_dict = {
    'custard apple':"This looks like a Custard Apple to me.", 
    'dragon fruit':"This looks like a Dragon fuit to me.", 
    'durian':"This looks like a Durian to me.", 
    'jackfruit':"This looks like a Jack Fruit to me.", 
    'mangosteen':"This looks like a Mangosteen to me.", 
    'nam dok mai mango':"This looks like a Nam Dok Mai Mango to me.", 
    'pomelo':"This looks like a Pomelo to me.", 
    'rambutan':"This looks like a Rambutan to me.", 
    'rose apple':"This looks like a Rose Apple to me.", 
    'salak fruit':"This looks like a Salak Fruit to me.", 
    'sapodilla':"This looks like a Sapodilla to me.", 
    'tamarind':"This looks like a Tamarind to me.", 
    'thai bananas':"This looks like a Thai Banana to me."
    }

#####################################
# Load model 
#####################################

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_model():
    return load_learner("new2.pkl")

with st.spinner("I am collecting my thoughts..."):
    model = load_model()
    
#####################################
# Upload image and make inference 
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
         - For each fruit type, I have been provided around 100 images to learn from
         - After 4 training runs (epochs), this was the result on the validation set:    
     """)
     st.image("images/confusion_matrix.png")
     st.markdown("""
         Want to know more?
         [Check out this Blog Series](https://ibronko.hashnode.dev/series/fast-ai)   
     """)
     
if st.button("Press button to load example image"):
    example_image = "images/example_image.jpg" 
    st.image(example_image)
    pred,pred_idx,probs = model.predict(example_image)
   
    st.success(f"{fruit_dict[pred]} I am {probs[pred_idx]*100:.0f}% confident.")
