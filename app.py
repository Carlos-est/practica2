import streamlit as st
import requests  
import json  
import numpy as np
import pandas as pd
import utils as ut
from PIL import Image
import time
# Título de la aplicación
st.title('PRACTICA 2')
if 'imagen_generada' not in st.session_state:
    # Si no existe, inicializarla
    st.session_state['imagen_generada'] = None
    #print("variable de estado inicializada")
## agregamos col1 a streamlib
col1, col2= st.columns(2)
with col1:
    st.subheader('Generation of images from text')
    input_text = st.text_input('Enter a text')
    # boton
    if st.button('Generar imagen'):
        result_link = ut.generacion_imagenes(input_text)
        time.sleep(80)
        #result_link = "https://pub-3626123a908346a7a8be8d9295f44e26.r2.dev/generations/0-0bef8989-5886-4882-b47c-ab4ae0db581e.png"
        # agregamos a variable de estado
        st.session_state['imagen_generada'] = result_link
        #   print("result_link: ",result_link, type(result_link))
        st.image(result_link, caption='Image - Shown', use_column_width=True)
        #st.session_state['imagen_generada'] = result_link
        st.write('AI generated image classification')
    if st.button('Classify generated image'):
        # guardamos en variable la variable de estado
        result_link_estado = st.session_state['imagen_generada']
        # si mi variable de estado es None
        if result_link_estado is not None:
            st.image(result_link_estado, caption='Image Shown', use_column_width=True)
            #result_ = ut.imagen_IA_para_clasificar(result_link_estado)
            result_ = ut.clasificacion_imagenes(result_link_estado)
            #print("result_: ",result_)
            st.markdown("**Classification result**")
            st.markdown("<div style='text-align: center; color: black; background-color: yellow; padding: 10px; border-radius: 5px;'><br>{}</div>".format(result_), unsafe_allow_html=True)
            
        else:
            st.write("The image has not been generated")
     

with col2:
    st.subheader('Image classification')
    uploaded_file = st.file_uploader("Choose a file")
    #print("uploaded_file: ",uploaded_file)
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Image Shown')
        if st.button('Classify uploaded image'):
                result = ut.clasificacion_imagenes(uploaded_file)
                st.markdown("**Classification result**")
                st.markdown("<div style='text-align: center; color: black; background-color: yellow; padding: 10px; border-radius: 5px;'><br>{}</div>".format(result), unsafe_allow_html=True)
