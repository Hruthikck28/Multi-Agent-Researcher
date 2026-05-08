from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from tools import web_search, search_arxiv, execute_python
import os
from dotenv import load_dotenv

load_dotenv()

print("Recruiting the AI Startup Intelligence Team...")

# 1. The Shared Brain
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

# 2. The Researcher Persona (Clean and simple)
researcher_prompt = """You are an elite Business Intelligence Researcher specializing in AI trends.
Use your tools to search the web for the most up-to-date information on the given topic.
Once you have retrieved the data using your tools, synthesize it into a final intelligence briefing.
Always cite your sources with URLs.
"""

researcher_agent = create_react_agent(
    model=llm,
    tools=[web_search, search_arxiv],
    prompt=researcher_prompt  
)

# 3. The Coder Persona
coder_prompt = """You are a Quantitative Data Analyst Agent.
Write Python code to calculate startup valuations, market growth rates, or perform data transformations.
Use execute_python to run your code and verify the output. 
Return the mathematical conclusions and the code used.
"""

coder_agent = create_react_agent(
    model=llm,
    tools=[execute_python],
    prompt=coder_prompt  
)

if __name__ == "__main__":
    test_query = "What is the recent funding news regarding the AI startup Anthropic?"
    print(f"\nDispatching Researcher to investigate: '{test_query}'...\n")
    
    result = researcher_agent.invoke({"messages": [("human", test_query)]})
    
    print("\n--- FINAL INTELLIGENCE REPORT ---")
    print(result["messages"][-1].content)