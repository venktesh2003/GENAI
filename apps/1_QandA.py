from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(
    
    model="gemini-3.8-flash"
)

st.title("Ask me buddy")
st.markdown("My chatbot using Streamlit")

query = st.chat_input("Ask anything?")

#use the session state so that your chat is preseved untill it is refreshed
if  "messages" not in st.session_state:
    st.session_state.messages = []  #messages is the variable

for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message("role").markdown(content)





if query:
    st.session_state.messages.append({"role":"user" , "content":query})  #we are storing this in messages so that our chat is presereved
    st.chat_message("user").markdown(query) #this one to dispy the question asked by user

    response = llm.invoke(query)

    st.chat_message("assistant").markdown(response.content)   #this one to display the ans by AI
    st.session_state.messages.append({"role":"ai" , "content":response})