import openai
import streamlit as st
from streamlit_feedback import streamlit_feedback
import trubrics
import fcntl
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

st.title("📝 User Feedback")

### 10/24 - edited
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #fcb290;
    }
</style>
""", unsafe_allow_html=True)

st.session_state.mainpageId = "True"

feedback_rate = streamlit_feedback(
        feedback_type="faces",
        align = "flex-start"
    )

user_query = st.text_input("User Feedback", key="user feedback")
st.session_state.data = []
# if not feedback_rate is None:
#     st.session_state.data.append(feedback_rate)
submit_button = st.button("Submit", type="primary")
if submit_button:
    # st.session_state.data.append(feedback_rate['score'])
    with open('./Customer_Review.txt', 'a', encoding='utf-8') as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        # if not feedback_rate is None:
        # st.write(st.session_state.data)
        # file.write('User rate: ' + st.session_state.data[0] + '\n')
        # print(feedback_rate)
        if not user_query is None:
            file.write('User review: ' + user_query + '\n')

