import gradio as gr
from workflow import app as agent_workflow

print("Launching the Premium AI Research Team Web Interface...")

def run_research(topic):
    # Yield initial status
    yield "### 🧠 Planner is breaking down the topic...\n\n*🕵️‍♂️ Researcher is checking memory and searching the web (this may take 10-20 seconds)...*"
    
    try:
        # Run the autonomous graph
        final_state = agent_workflow.invoke({"topic": topic, "messages": []})
        
        # Extract the final Markdown report
        report = final_state["final_report"]
        
        # Yield the final result to the UI
        yield report
        
    except Exception as e:
        yield f"❌ **An error occurred during research:**\n\n`{str(e)}`"

# Inject some custom CSS to make the report container look like a premium document card
custom_css = """
.report-container {
    padding: 30px;
    background-color: #f8fafc;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.dark .report-container {
    background-color: #1e293b;
    border: 1px solid #334155;
}
"""

# Build the Gradio Web Interface using a cleaner, search-engine style layout
with gr.Blocks(theme=gr.themes.Default(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as demo:
    
    # 1. Centered Header
    with gr.Column(elem_id="header"):
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 0px;'>🌐 Autonomous Business Intelligence</h1>")
        gr.Markdown("<p style='text-align: center; color: gray; margin-top: 5px; font-size: 16px;'>Enter a tech trend, startup, or market question. The AI team will plan, search, recall, and synthesize.</p>")
    
    # 2. Sleek Search Bar Row
    with gr.Row(equal_height=True):
        topic_input = gr.Textbox(
            show_label=False,
            placeholder="e.g., The rise of humanoid robots in manufacturing 2024...",
            scale=4, # Takes up 80% of the row
            container=False # Removes the bulky box outline
        )
        submit_btn = gr.Button("🚀 Dispatch AI Team", variant="primary", scale=1) # Takes up 20%
        
    # 3. Full-Width Report Display Area
    with gr.Column(elem_classes="report-container"):
        output_display = gr.Markdown("### 📊 Executive Report will appear here...")

    # 4. Triggers (Clicking the button OR pressing Enter)
    submit_btn.click(fn=run_research, inputs=topic_input, outputs=output_display)
    topic_input.submit(fn=run_research, inputs=topic_input, outputs=output_display)

if __name__ == "__main__":
    # Launch the web server
    demo.launch()