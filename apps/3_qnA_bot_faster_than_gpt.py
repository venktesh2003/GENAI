#LLM,  TOOL , AGENT , MEMORY , STREAMING , WEB Interface
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st   
#creating the LLl

llm = ChatGroq(model="openai/gpt-oss-20b" , streaming=True)
search = GoogleSerperAPIWrapper()
tools=[search.run]
#create memort for our agent

#so to keep when we are asking quesrion it is again creating new object of memory so we will create memory that will rember it
if "memory" not in st.session_state:
   st.session_state.memory=MemorySaver()
   st.session_state.history=[]
   
memory=st.session_state.memory
agent = create_agent(model=llm , tools=tools , checkpointer=memory , system_prompt="you are amazing search agent and can search on google ")

#Building web Interface

st.subheader("Quick answer chatbot -- speed of flash")
for message in st.session_state.history:
   role=message["role"]
   content=message["content"]
   st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything ? ")
if query:
   st.chat_message("user").markdown(query)
   st.session_state.history.append({"role":"user" , "content":query})

   res=agent.stream({"messages":[{"role":"user" , "content":query}]} , 
                 
                 {"configurable":{"thread_id":1}} ,
                 stream_mode="messages" )
# displaying the AI response by creating container in CHAT APP

   ai_container=st.chat_message("ai")
   with ai_container:
      space = st.empty()
      message = ""

      for chunk in res:
         message =  message + chunk[0].content
         space.write(message)
         
         
      st.session_state.history.append({"role":"ai" , "content":message})   
      #    st.session_state.history.append({"role":"AI" , "content":res["messages"][-1].content})
#    st.chat_message("AI").markdown(res["messages"][-1].content)


    


























