import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import pgeocode
import pandas as pd
import openai
import re
import requests
from PIL import Image
from steamship import Steamship
from datetime import datetime
import numpy as np
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from datetime import datetime, time
from branca.element import IFrame
# pip install "trubrics[streamlit]" for feedback
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback

st.title("ChatGPT-like clone")

api_key = 'sk-None-F4ump4p48TwdpLJ8zPDLT3BlbkFJRbUZiveN9s2NIBKlO3GF'
openai.api_key = api_key

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.chat_message("assistant"):
    prompt = 'When is Temple University?'
    stream = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=True,
    )
    response = st.write_stream(stream)
st.session_state.messages.append({"role": "assistant", "content": response})