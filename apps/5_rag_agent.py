from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

########### store data in session ##########################\

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent=None

if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "messages" not in st.session_state:
    st.session_state.messages=[]







def process_document(path):



    ######## load the documents ####################
    loader = PyPDFDirectoryLoader(path)
    docs=loader.load()

    ########split the documents into chunks #################
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    docs=splitter.split_documents(documents=docs)

    ########### embeedings and vector stores ############################
    embd=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db=InMemoryVectorStore(
        
        embedding=embd
    )
    vector_db.add_documents(docs)

    ############## create agent #######################
    # for agent we need tool , sysprompt openai

    llm= ChatGroq(model="openai/gpt-oss-20b")

    ##### creating tool #################
    @tool
    def retrive_context(query:str):
        '''
        retrieve documents relevant to query from knowledege base 
        '''
        context=''
        docs=vector_db.similarity_search(query=query , k=3)
        for doc in docs:
            context= context + doc.page_content + "\n\n"
            print("query" , query)
        
        return context

    ######## create sys prompt ##################################
    sys_prm=""" you are a good assistant agent answer the questions from my knowledege base and detail from uplodate document Always use the retrieve_context tool for returning the answer"""

    #######create memory obj #################################

    memory = InMemorySaver()

    ####### create agent #################################
    agent = create_agent(model=llm , tools=[retrive_context] , system_prompt=sys_prm , checkpointer=memory)
    st.session_state.agent=agent
    st.session_state.document_uploaded = True


###########upload UI ##############################################

# we will show upload section untill  we don have the document uploaded

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="select the pdf files " , type=["pdf"] , accept_multiple_files=True)
    if uploaded:
        with st.spinner("processing ......."):
            path="./apps/doc_files/"
            for file in uploaded:
                with open(path+file.name , "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()   #once doc is uploaded and proceesed you need another window so we use this one to refresh the data

####################### chat UI ##########################################################


if  st.session_state.document_uploaded and st.session_state.agent:
    query=st.chat_input("ask me anything related to uploaded document")
    for message in st.session_state.messages:
        role=message.get("role")
        content=message.get("content")
        st.chat_message(role).markdown(content)
    if query:
        st.session_state.messages.append({"role":"user" , "content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user" , "content":query}]} ,  {"configurable": {"thread_id": "user_123"}   }
        )
        ans=response["messages"][-1].content
        st.chat_message("AI").markdown(ans)
        st.session_state.messages.append({"role":"AI" , "content":ans})








# ###### asking qustio to use and building chat
# while True:
#     query=input("User: ")
#     if query.lower()=="quit":
#         print("good bye")
#         break
#     response=agent.invoke({"messages":[{"role":"user" , "content":query}] } , {"configurable":{"thread_id":1}})
#     result=response["messages"][-1].content
#     print("AI" , result)






























































































































































































































































































































































































































































































































































































































































