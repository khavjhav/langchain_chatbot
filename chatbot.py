from urllib import response
import streamlit as st
import requests
from assistant import chat_generator
st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    pass

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = requests.post("http://localhost:8001/chat",{"messages":st.session_state.messages})

    # msg = response.choices[0].message.content
    # print(st.session_state.messages)
    msg=chat_generator(st.session_state.messages)
    # msg="gg"
    print(type({"messages":st.session_state.messages}))
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
