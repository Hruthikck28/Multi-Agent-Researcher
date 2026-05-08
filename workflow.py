import json
import time
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import our agents AND our new memory functions
from agents import llm, researcher_agent
from memory import store_finding, recall_relevant

print("Initializing LangGraph AI Orchestrator (with Persistent Memory)...")

class ResearchState(TypedDict):
    topic: str
    plan: dict
    findings: str
    final_report: str
    messages: Annotated[list, add_messages]

def planner_node(state: ResearchState):
    print(f"\n🧠 [Planner] Breaking down the topic: '{state['topic']}'...")
    prompt = f"""You are a Tech Research Director. Break down this topic into 2 specific Google search queries to gather intelligence.
    Topic: {state['topic']}
    Return ONLY valid JSON in this exact format: {{"queries": ["search query 1", "search query 2"]}}
    """
    
    response = llm.invoke(prompt)
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    plan = json.loads(clean_json)
    return {"plan": plan}

def researcher_node(state: ResearchState):
    print("\n🕵️‍♂️ [Researcher] Gathering intelligence...")
    all_findings = ""
    
    for query in state["plan"]["queries"]:
        print(f"\n   -> Investigating: {query}")
        
        # 1. Check Long-Term Memory First
        # We pull the distance score to ensure it's actually a highly relevant memory, not just a random guess
        embedding = recall_relevant(query, n_results=1) 
        
        # Note: Because our simple recall_relevant returns a list of strings, we'll do a basic keyword check 
        # to ensure the memory is actually about our target topic before trusting it.
        target_keyword = state['topic'].split()[0].lower()
        
        if embedding and target_keyword in embedding[0].lower():
            print("      [🧠 MEMORY CACHE HIT] I already know this! Skipping web search...")
            all_findings += embedding[0] + "\n\n"
            continue
            
        # 2. If not in memory (or not relevant), hit the live internet
        print("      [🌐 MEMORY MISS] Searching the live internet...")
        result = researcher_agent.invoke({"messages": [("human", query)]})
        new_finding = result["messages"][-1].content
        
        # 3. Save the new knowledge to ChromaDB for the future
        print("      [💾 SAVING TO MEMORY] Committing new findings to Vector DB...")
        store_finding(new_finding, state["topic"])
        
        all_findings += new_finding + "\n\n"
        
        # Throttle to respect API limits
        print("      [⏳] Pausing for 3 seconds...")
        time.sleep(3) 
        
    return {"findings": all_findings}

def writer_node(state: ResearchState):
    print("\n✍️ [Writer] Synthesizing final business report...")
    prompt = f"""You are an elite Tech Market Analyst. Write a professional, structured executive summary based ONLY on these findings:
    
    FINDINGS:
    {state['findings']}
    
    Format the output in clean Markdown. Include a "Market Overview" section, "Key Players", and a "Strategic Takeaways" bulleted list.
    """
    
    response = llm.invoke(prompt)
    return {"final_report": response.content}

# Build the Graph
workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

if __name__ == "__main__":
    # We will test the memory caching with a specific startup
    target_topic = "OpenAI latest funding and valuation 2024"
    
    print("\n🚀 STARTING AUTONOMOUS RESEARCH RUN")
    final_state = app.invoke({"topic": target_topic, "messages": []})
    
    print("\n" + "="*60)
    print("📊 FINAL EXECUTIVE REPORT")
    print("="*60 + "\n")
    print(final_state["final_report"])