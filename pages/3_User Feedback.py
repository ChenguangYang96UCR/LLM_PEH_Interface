import openai
import streamlit as st
from streamlit_feedback import streamlit_feedback
import trubrics
#from langchain.agents import AgentType
#from langchain_experimental.agents import create_pandas_dataframe_agent
#from langchain.callbacks import StreamlitCallbackHandler
#from langchain.chat_models import ChatOpenAI

#pip install trubrics
#pip install langchain-experimental


#with st.sidebar:
#    openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")
#    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/5_Chat_with_user_feedback.py)"
#    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("📝 Chat with Feedback")

### 10/24 - edited
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #fcb290;
    }
</style>
""", unsafe_allow_html=True)


#"""
#In this example, we're using [streamlit-feedback](https://github.com/trubrics/streamlit-feedback) and Trubrics to collect and store feedback
#from the user about the LLM responses.
#"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you? Leave feedback to help DREAM-KG chatbot improve!"}
    ]
if "response" not in st.session_state:
    st.session_state["response"] = None

messages = st.session_state.messages
for msg in messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #prompt = "You are an expert on social science and homeless population analysis in Philadelphia and your job is answer end-users' questions."
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()
    #client = OpenAI(api_key=openai_api_key)
    api_key = 'sk-tmybDmPl6ensfuOzjDwcEDDYg9AFVlUyg4pfUasyKrT3BlbkFJu_s1RvLGnpg-p34GjicHzYLhfmQVDud0XJuPsOyKsA'
    openai.api_key = api_key
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    st.session_state["response"] = response.choices[0].message.content
    with st.chat_message("assistant"):
        messages.append({"role": "assistant", "content": st.session_state["response"]})
        st.write(st.session_state["response"])

if st.session_state["response"]:
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{len(messages)}",
    )
    # This app is logging feedback to Trubrics backend, but you can send it anywhere.
    # The return value of streamlit_feedback() is just a dict.
    # Configure your own account at https://trubrics.streamlit.app/
    if feedback and "TRUBRICS_EMAIL" in st.secrets:
        config = trubrics.init(
            email=st.secrets.TRUBRICS_EMAIL,
            password=st.secrets.TRUBRICS_PASSWORD,
        )
        collection = trubrics.collect(
            component_name="default",
            model="gpt",
            response=feedback,
            metadata={"chat": messages},
        )
        trubrics.save(config, collection)
        st.toast("Feedback recorded!", icon="📝")