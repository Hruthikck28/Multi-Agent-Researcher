import time
import json
from langchain_core.prompts import ChatPromptTemplate
from workflow import app as agent_workflow
from agents import llm

print("Initializing Phase 6: LLM-as-a-Judge Benchmarking Pipeline...")

# 1. Define our Ground Truth Test Suite
# We give the judge specific facts that MUST be in the report to score highly
benchmark_tests = [
    {
        "query": "What is the recent funding news and valuation for the AI startup Anthropic in 2024?",
        "expected_facts": "Must mention $900 billion valuation rumors, recent $30B+ funding history, and backers like Amazon or Google."
    },
    {
        "query": "What is NVIDIA's current market share in specialized AI chips?",
        "expected_facts": "Must mention they hold a dominant majority share (often cited around 80%+), and mention growing competition from AMD."
    }
]

# 2. Build the LLM Judge Prompt
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an impartial, strict AI grading system. 
Rate the following research report on a scale of 1-10 for three metrics:
1. Accuracy: Does it contain the 'expected facts'?
2. Completeness: Does it fully answer the original query?
3. Depth: Is the report professional, structured, and insightful?

Query asked: {query}
Expected Facts: {expected_facts}

Report to grade:
{report}

Return ONLY valid JSON in this exact format:
{{"accuracy": 8, "completeness": 9, "depth": 7, "feedback": "Short explanation of why."}}
""")
])

# 3. Run the Evaluation Loop
print("\n🚀 Starting Evaluation Run (This will take a minute to process multiple queries)...\n")

total_score = 0
total_time = 0

for i, test in enumerate(benchmark_tests):
    print(f"--- Running Test {i+1}/2: {test['query'][:40]}... ---")
    
    start_time = time.time()
    
    # Run the autonomous graph
    final_state = agent_workflow.invoke({"topic": test['query'], "messages": []})
    report = final_state["final_report"]
    
    end_time = time.time()
    latency = round(end_time - start_time, 2)
    total_time += latency
    
    print(f"✅ Report generated in {latency} seconds. Handing to LLM Judge...")
    
    # Pass the report to the LLM Judge
    eval_chain = judge_prompt | llm
    eval_response = eval_chain.invoke({
        "query": test["query"],
        "expected_facts": test["expected_facts"],
        "report": report
    })
    
    # Clean and parse the JSON score
    clean_json = eval_response.content.replace("```json", "").replace("```", "").strip()
    try:
        score = json.loads(clean_json)
        avg_score = round((score["accuracy"] + score["completeness"] + score["depth"]) / 3, 1)
        total_score += avg_score
        
        print("\n📊 JUDGE'S SCORECARD:")
        print(f"Accuracy:     {score['accuracy']}/10")
        print(f"Completeness: {score['completeness']}/10")
        print(f"Depth:        {score['depth']}/10")
        print(f"Feedback:     {score['feedback']}\n")
    except Exception as e:
        print(f"Error parsing judge score: {e}\nRaw Output: {eval_response.content}")

# Print Final Benchmark Results
print("="*50)
print("🏆 FINAL SYSTEM BENCHMARK RESULTS")
print("="*50)
print(f"Average Report Quality Score: {total_score / len(benchmark_tests)} / 10")
print(f"Average System Latency:       {total_time / len(benchmark_tests)} seconds per report")
print("="*50)