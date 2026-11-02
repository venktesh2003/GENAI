from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
llm=ChatGroq(model="openai/gpt-oss-20b")  #llm
search=GoogleSerperAPIWrapper()   #search agent
# create agent now 
# agent=create_agent(model=llm , 
# tools=[search.run] , 
# system_prompt="your are agent and you can search question on google"
# )

#############adding memory to our agent #####################
agent=create_agent(model=llm , 
tools=[search.run] , 
system_prompt="your are agent and you can search question on google" ,
checkpointer=InMemorySaver()
)

while True:
    query=input("User: ")
    if query.lower()=="quit":
        print("good bye")
        break
    res=agent.invoke({"messages":[{"role":"user" , "content":query}]} ,{"configurable":{"thread_id":1}})
    print("AI: " , res["messages"][-1].content)
