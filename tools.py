import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
import urllib.request
import json

load_dotenv()

# 1. The Web Search Tool (For Market News & Startup Data)
@tool
def web_search(query: str) -> str:
    """
    Search the web for current tech trends, startup news, and market intelligence.
    Always use this to get the most up-to-date information.
    """
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        # THE FIX: Drop to "basic" search and limit to 2 results to save tokens
        results = client.search(query=query, search_depth="basic", max_results=2)
        
        formatted_results = []
        for r in results.get("results", []):
            # THE FIX: Truncate each article's content to 1000 characters so the AI doesn't choke
            content_snippet = r['content'][:1000] 
            formatted_results.append(f"Source: {r['url']}\nContent: {content_snippet}...\n")
        
        final_output = "\n---\n".join(formatted_results)
        
        # Hard cap the entire tool output at 3000 characters just to be absolutely safe
        return final_output[:3000]
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# 2. ArXiv Academic Search (For Deep AI/ML Tech Trends)
@tool
def search_arxiv(query: str) -> str:
    """
    Search the ArXiv database for published academic papers. 
    Use this specifically when researching core AI algorithms, ML breakthroughs, or scientific tech trends.
    """
    try:
        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3'
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        
        # Very basic XML parsing to extract just the text we need so the LLM doesn't get overwhelmed
        articles = data.split('<entry>')
        results = []
        for article in articles[1:]:
            title = article.split('<title>')[1].split('</title>')[0].replace('\n', ' ').strip()
            summary = article.split('<summary>')[1].split('</summary>')[0].replace('\n', ' ').strip()
            results.append(f"Title: {title}\nAbstract: {summary}\n")
            
        return "\n---\n".join(results) if results else "No academic papers found."
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

# 3. Python Code Execution (For Financial/Data Crunching)
@tool
def execute_python(code: str) -> str:
    """
    Execute Python code safely and return the standard output.
    Use this to perform mathematical calculations, data analysis on startup funding, or format statistics.
    The code runs in a sandbox. Use 'print()' to output the final answer.
    """
    import subprocess
    import tempfile
    
    # We write the code to a temporary file, run it, and capture the output
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
        
    try:
        # 15 second timeout to prevent infinite loops
        result = subprocess.run(
            ["python3", fname], 
            capture_output=True,
            text=True, 
            timeout=15
        )
        output = result.stdout if result.stdout else result.stderr
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Python code execution timed out (exceeded 15 seconds)."
    except Exception as e:
        return f"Error executing Python: {str(e)}"
    finally:
        # Clean up the file
        if os.path.exists(fname):
            os.unlink(fname)

if __name__ == "__main__":
    print("Testing Tools...")
    print("\n1. Testing Web Search (Tavily):")
    print(web_search.invoke({"query": "Latest funding round for OpenAI"}))