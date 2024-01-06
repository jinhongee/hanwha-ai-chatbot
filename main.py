from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

prompts = [
    "You can try one of these prompts:\n\n"
    "What were the plans to combat inflation?",
    "How did President Biden describe the American Rescue Plan?",
    "What is described as prevailing over tyranny in the speech?", 
    "What is mentioned as the cause for the distance between people last year?",
    "Which leader is accused of misjudging the global response to their actions?",
    "Which country and its people are commended for their resistance and bravery?",
    "What did President Biden propose regarding energy and child care costs in his 2022 State of the Union address?"
]

concatenated_string = "\n".join(prompts)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Nice to meet you! I am Hanhwa's AI bot trained on Biden's State of the Union Speech", concatenated_string]

if 'requests' not in st.session_state:
    st.session_state['requests'] = ["Hello, nice to meet you!"]

llm = ChatOpenAI(model_name="gpt-4", openai_api_key="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")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


st.title("Hanhwa AI Chatbot\n")
st.write('<br><br>', unsafe_allow_html=True)


response_container = st.container()
textcontainer = st.container()
st.write('<br><br>', unsafe_allow_html=True)

def clear_text():
    st.session_state["input"] = ""
    context = find_match(query)
    response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
    st.session_state.requests.append(query)
    st.session_state.responses.append(response) 

hide_label_style = """
    <style>
    .label {display: none;}
    </style>
    """
st.markdown(hide_label_style, unsafe_allow_html=True)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

with textcontainer:
    st.write('<br><br>', unsafe_allow_html=True)
    col1, col2 = st.columns([7, 1], gap="small")  # Adjust the ratio and gap as needed

    with col1:
        query = st.text_input("", key="input", placeholder="Start by saying hello!", label_visibility="collapsed")

    with col2:
        st.button("Search", on_click=clear_text)

