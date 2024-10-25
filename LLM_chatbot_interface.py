import openai
import streamlit as st

st.title("ChatGPT-like clone")

openai.api_key = 'sk-None-F4ump4p48TwdpLJ8zPDLT3BlbkFJRbUZiveN9s2NIBKlO3GF'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
