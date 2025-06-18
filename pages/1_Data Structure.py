import openai
import streamlit as st
from streamlit_feedback import streamlit_feedback
import trubrics
import fcntl
from PIL import Image


### 10/24 - edited
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #1C1C1C;
    }
</style>
""", unsafe_allow_html=True)

data_figure = Image.open('./figures/dreamkg_data_figure.png')
st.image(data_figure,caption= 'DREAM-KG Data Resources.')

st.write('\n')
data_struct = Image.open('./figures/data_base.png')
st.image(data_struct,caption= 'DREAM-KG Schema.')
