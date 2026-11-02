from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()
llm=ChatGroq(model="openai/gpt-oss-20b")  #llm
search=GoogleSerperAPIWrapper()   #search agent
# create agent now 
agent=create_agent(model=llm , 
tools=[search.run] , 
system_prompt="your are agent and you can search question on google"
)

while True:
    query=input("User: ")
    if query.lower()=="quit":
        print("good bye")
        break
    res=agent.invoke({"messages":[{"role":"user" , "content":query}]})
    print("AI: " , res["messages"][-1].content)
