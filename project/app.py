from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, AzureMLEndpointApiType, CustomOpenAIChatContentFormatter
import os
import urllib.parse
import pandas as pd
import re
import uuid
import sqlalchemy
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.language_models.llms import LLM
from typing import Optional, TypedDict, Any
from pydantic import Field
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from langgraph.graph import StateGraph, END

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class AppState(TypedDict, total=False):
    question: str
    sql_query: str
    sql_result: Any
    df: pd.DataFrame
    viz_path: str
    answer: str

llm =  AzureMLChatOnlineEndpoint(
    endpoint_url="https://Meta-Llama-3-1-70B-Instruct-harz.westus.models.ai.azure.com/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key="FvK3Y5jqH5Z8IacNhgJw1UQh2QFkcVoJ",
    content_formatter=CustomOpenAIChatContentFormatter(),
    model_kwargs={"temperature": 0.3, "max_tokens": 2000}
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

def setup_database_connection():
    user = "postgres"
    db_password = "Qwerty@532"
    host = "localhost"
    port = "5432"
    db = "532"

    encoded_password = urllib.parse.quote(db_password, safe="")
    url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
    engine = sqlalchemy.create_engine(url)
    try:
        db = SQLDatabase(engine, schema="analytical_schema")
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to establish database connection: {str(e)}")

# Initialize database connection
db = setup_database_connection()

# def sql_node(state):
#     question = state["question"]
#     sql_prompt = f"""Generate a SQL query to answer: {question}. 
#     The database has these tables:\n\n{db.get_table_info()}
#     Return only the SQL query, nothing else."""
#     sql_query = conversation.predict(input=sql_prompt).strip()
#     sql_query = re.sub(r'```sql|```', '', sql_query).strip()
#     result = QuerySQLDatabaseTool(db=db).invoke(sql_query)
#     df = pd.read_sql(sql_query, db._engine)

#     print("\n--- SQL QUERY ---\n", sql_query)
#     print("\n--- SQL RESULT PREVIEW ---\n", df.head(5))

#     return {**state, "sql_query": sql_query, "sql_result": result, "df": df}

# def sql_node(state):
#     question = state["question"]
#     sql_prompt = f"""Generate a SQL query to answer: {question}. 
#     The database has these tables:\n\n{db.get_table_info()}
#     Return only the SQL query, nothing else."""
    
#     sql_query = conversation.predict(input=sql_prompt).strip()
#     sql_query = re.sub(r'```sql|```', '', sql_query).strip()

#     try:
#         result = QuerySQLDatabaseTool(db=db).invoke(sql_query)
#         df = pd.read_sql(sql_query, db._engine)

#         print("\n--- SQL QUERY ---\n", sql_query)
#         print("\n--- SQL RESULT PREVIEW ---\n", df.head(5))

#         return {**state, "sql_query": sql_query, "sql_result": result, "df": df, "retry_count": 0}
#     except Exception as e:
#         print("[ERROR] SQL execution failed:", str(e))
#         return {**state, "sql_query": sql_query, "sql_error": str(e), "retry_count": 0}

def sql_node(state):
    print("\n=== [NODE] sql_agent ===")
    print("[INPUT STATE]:", state)

    question = state["question"]
    sql_prompt = f"""Generate a SQL query to answer: {question}. 
    The database has these tables:\n\n{db.get_table_info()}
    Return only the SQL query, nothing else."""
    
    sql_query = conversation.predict(input=sql_prompt).strip()
    sql_query = re.sub(r'```sql|```', '', sql_query).strip()

    try:
        result = QuerySQLDatabaseTool(db=db).invoke(sql_query)
        df = pd.read_sql(sql_query, db._engine)

        updated_state = {**state, "sql_query": sql_query, "sql_result": result, "df": df, "retry_count": 0}
        print("[OUTPUT STATE]:", updated_state)
        return updated_state
    except Exception as e:
        updated_state = {**state, "sql_query": sql_query, "sql_error": str(e), "retry_count": 0}
        print("[OUTPUT STATE]:", updated_state)
        return updated_state

# def sql_checker_node(state):
#     retry_count = state.get("retry_count", 0)
#     if retry_count >= 3:
#         print("[ERROR] Retry limit reached.")
#         return state

#     question = state["question"]
#     faulty_query = state.get("sql_query", "")
#     error_msg = state.get("sql_error", "")
#     table_info = db.get_table_info()

#     checker_prompt = f"""
# You are a SQL expert helping fix queries for a question-answering system.

# Given:
# - User Question: {question}
# - SQL Query: {faulty_query}
# - Table Information: {table_info}
# - Error Message: {error_msg}

# Your task:
# 1. Identify and fix the SQL query error based on the error message.
# 2. Return ONLY the corrected SQL query.

# Correct SQL Query:"""

#     corrected_query = conversation.predict(input=checker_prompt).strip()
#     corrected_query = re.sub(r'```sql|```', '', corrected_query).strip()

#     return {**state, "sql_query": corrected_query, "sql_error": None, "retry_count": retry_count + 1}

def sql_checker_node(state):
    print("\n=== [NODE] sql_checker ===")
    print("[INPUT STATE]:", state)

    retry_count = state.get("retry_count", 0)
    if retry_count >= 3:
        print("[ERROR] Retry limit reached.")
        return state

    question = state["question"]
    faulty_query = state.get("sql_query", "")
    error_msg = state.get("sql_error", "")
    table_info = db.get_table_info()

    checker_prompt = f"""
You are a SQL expert helping fix queries for a question-answering system.

Given:
- User Question: {question}
- SQL Query: {faulty_query}
- Table Information: {table_info}
- Error Message: {error_msg}

Your task:
1. Identify and fix the SQL query error based on the error message.
2. Return ONLY the corrected SQL query.

Correct SQL Query:"""

    corrected_query = conversation.predict(input=checker_prompt).strip()
    corrected_query = re.sub(r'```sql|```', '', corrected_query).strip()

    updated_state = {**state, "sql_query": corrected_query, "sql_error": None, "retry_count": retry_count + 1}
    print("[OUTPUT STATE]:", updated_state)
    return updated_state


# def generate_visualization(df, question, query):
#     print("\n[DEBUG] Generating visualization for:", question)
#     print("[DEBUG] SQL Query:", query)
#     print("[DEBUG] DataFrame Head:\n", df.head())

#     df_description = f"""
#     User Question: {question}
#     SQL Query: {query}
#     Resulting Data:
#     - Number of rows: {len(df)}
#     - Number of columns: {len(df.columns)}
#     - Columns: {', '.join(df.columns)}
#     {df.to_string()}
#     Data Types:
#     {df.dtypes.to_string()}
#     """

#     prompt = f"""You are a data visualization expert. Analyze the following SQL query results and recommend the most appropriate visualization.

#     {df_description}

#     Respond with:
#     1. The recommended visualization type (bar, line, pie, scatter, histogram, stacked bar graph etc.)
#     2. The columns to use for x-axis, y-axis, etc.
#     3. A suggested title
#     4. A brief explanation of why this visualization fits

#     Format your response as:
#     Visualization Type: <type>
#     X-Axis: <column or none>
#     Y-Axis: <column or none>
#     Color: <column or none>
#     Title: <suggested title>
#     Explanation: <brief explanation>"""

#     recommendation = conversation.predict(input=prompt).strip()
#     print("[DEBUG] Visualization Recommendation:\n", recommendation)

#     plotly_prompt = f"""Create Plotly visualization code in Python based on these specifications:
#     Data Sample:
#     {df.to_string()}
#     Visualization Specifications:
#     {recommendation}

#     Generate complete Python code using Plotly that:
#     1. Creates the visualization
#     2. Includes proper axis labels and title
#     3. Handles any necessary data transformations 
#     - Use a pandas DataFrame called 'df'
#     - Use plotly.graph_objects (go) or plotly.express (px)
#     - Dont include fig.show() function in the code
#     - No markdown or code blocks
#     - No explanations or comments"""

#     plotly_code = conversation.predict(input=plotly_prompt).strip()
#     print("[DEBUG] Generated Plotly Code:\n", plotly_code)

#     if '```' in plotly_code:
#         plotly_code = re.split(r'```(?:python)?', plotly_code)[-1].split('```')[0]

#     local_vars = {'df': df, 'go': go, 'px': px, 'pd': pd}
#     try:
#         exec(plotly_code, globals(), local_vars)
#         fig = None
#         for var in local_vars.values():
#             if isinstance(var, go.Figure):
#                 fig = var
#                 break

#         if fig is None:
#             raise ValueError("Generated code did not produce a Plotly figure object")

#         filename = f"viz_{uuid.uuid4().hex}.png"
#         output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         fig.write_image(output_path)
#         print("[DEBUG] Visualization saved to:", output_path)
#         return filename

#     except Exception as e:
#         print(f"[ERROR] Visualization generation failed: {str(e)}")
#         return None

def generate_visualization(df, plotly_code):
    print("[DEBUG] Executing Plotly code to generate visualization...")

    # Remove markdown/code block artifacts if any
    if '```' in plotly_code:
        plotly_code = re.split(r'```(?:python)?', plotly_code)[-1].split('```')[0]

    local_vars = {'df': df, 'go': go, 'px': px, 'pd': pd}
    try:
        exec(plotly_code, globals(), local_vars)
        fig = None
        for var in local_vars.values():
            if isinstance(var, go.Figure):
                fig = var
                break

        if fig is None:
            raise ValueError("Generated code did not produce a Plotly figure object")

        filename = f"viz_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fig.write_image(output_path)
        print("[DEBUG] Visualization saved to:", output_path)
        return filename

    except Exception as e:
        print(f"[ERROR] Visualization generation failed: {str(e)}")
        return None

# def viz_node(state):
#     df = state["df"]
#     if df.empty:
#         return {**state, "viz_path": None}
#     fname = generate_visualization(df, state['question'], state['sql_query'])
#     return {**state, "viz_path": fname}

# def viz_node(state):
#     print("\n=== [NODE] viz_agent ===")
#     print("[INPUT STATE]:", state)

#     df = state["df"]
#     if df.empty:
#         updated_state = {**state, "viz_path": None}
#         print("[OUTPUT STATE]:", updated_state)
#         return updated_state

#     fname = generate_visualization(df, state['question'], state['sql_query'])
#     updated_state = {**state, "viz_path": fname}
#     print("[OUTPUT STATE]:", updated_state)
#     return updated_state

def viz_node(state):
    print("\n=== [NODE] viz_agent ===")
    print("[INPUT STATE]:", state)

    df = state["df"]

    if df.empty:
        print("[INFO] Empty DataFrame detected. Skipping visualization.")
        updated_state = {**state, "viz_path": None}
        print("[OUTPUT STATE]:", updated_state)
        return updated_state

    question = state['question']
    query = state['sql_query']

    print("\n[DEBUG] Generating visualization for:", question)
    print("[DEBUG] SQL Query:", query)
    print("[DEBUG] DataFrame Head:\n", df.head())

    df_description = f"""
    User Question: {question}
    SQL Query: {query}
    Resulting Data:
    - Number of rows: {len(df)}
    - Number of columns: {len(df.columns)}
    - Columns: {', '.join(df.columns)}
    {df.to_string()}
    Data Types:
    {df.dtypes.to_string()}
    """

    prompt = f"""You are a data visualization expert. Analyze the following SQL query results and recommend the most appropriate visualization.

    {df_description}

    If the data is not suitable for visualization (example: single value, redundant rows, empty, very few rows), then respond with exactly: "No visualization required."

    Otherwise, respond with:
    1. The recommended visualization type (bar, line, pie, scatter, histogram, stacked bar graph etc.)
    2. The columns to use for x-axis, y-axis, etc.
    3. A suggested title
    4. A brief explanation of why this visualization fits

    Format your response properly."""
    
    recommendation = conversation.predict(input=prompt).strip()
    print("[DEBUG] Visualization Recommendation:\n", recommendation)

    if "no visualization required" in recommendation.lower():
        print("[INFO] LLM decided no visualization is necessary.")
        updated_state = {**state, "viz_path": None}
        print("[OUTPUT STATE]:", updated_state)
        return updated_state

    # If visualization is required, generate Plotly code
    plotly_prompt = f"""Create Plotly visualization code in Python based on these specifications:
    Data Sample:
    {df.to_string()}
    Visualization Specifications:
    {recommendation}

    Generate complete Python code using Plotly that:
    - Creates the visualization
    - Includes proper axis labels and title
    - Handles any necessary data transformations 
    - Use a pandas DataFrame called 'df'
    - Use plotly.graph_objects (go) or plotly.express (px)
    - Don't include fig.show()
    - No markdown or code blocks
    - No explanations or comments"""

    plotly_code = conversation.predict(input=plotly_prompt).strip()
    print("[DEBUG] Generated Plotly Code:\n", plotly_code)

    fname = generate_visualization(df, plotly_code)
    updated_state = {**state, "viz_path": fname}
    print("[OUTPUT STATE]:", updated_state)
    return updated_state

# def final_summary_node(state):
#     prompt = f"""
# You are a smart analyst. Use the following context to summarize the answer to the user's question clearly and concisely.

# User Question:
# {state['question']}

# SQL Query Output:
# {state.get('sql_result', 'Not available')}

# {"A visualization was also generated to support the results." if state.get('viz_path') else "No visualization was created."}

# Write a final answer in plain English that:
# - Answers the user's question directly
# - Highlights key trends, numbers, or patterns
# - Mentions the chart if it's relevant

# Final Summary:"""
    
#     answer = conversation.predict(input=prompt).strip()
#     return {**state, "answer": answer}


# def supervisor_node(state):
#     return state

def final_summary_node(state):
    print("\n=== [NODE] final_summary ===")
    print("[INPUT STATE]:", state)

    prompt = f"""
You are a smart analyst. Use the following context to summarize the answer to the user's question clearly and concisely.

User Question:
{state['question']}

SQL Query Output:
{state.get('sql_result', 'Not available')}

{"A visualization was also generated to support the results." if state.get('viz_path') else "No visualization was created."}

Write a final answer in plain English that:
- Answers the user's question directly
- Highlights key trends, numbers, or patterns
- Mentions the chart if it's relevant

Final Summary:"""
    
    answer = conversation.predict(input=prompt).strip()
    updated_state = {**state, "answer": answer}
    print("[OUTPUT STATE]:", updated_state)
    return updated_state

def supervisor_node(state):
    print("\n=== [NODE] supervisor ===")
    print("[INPUT STATE]:", state)
    print("[OUTPUT STATE]:", state)
    return state

# def supervisor_router(state):
#     if "sql_query" not in state:
#         return "sql_agent"
#     elif "viz_path" not in state or not state["viz_path"]:
#         return "viz_agent"
#     else:
#         return "final_summary"

def print_conversation_memory(memory):
    """
    Pretty prints the current conversation memory:
    - Shows each turn with [User] or [AI] tags
    - Helps you see how big the memory is getting
    """
    print("\n=== [Conversation Memory Dump] ===")
    
    if not memory.chat_memory.messages:
        print("Memory is empty.")
        return

    for i, msg in enumerate(memory.chat_memory.messages):
        role = "User" if msg.type == "human" else "AI"
        print(f"[{i+1}] [{role}]: {msg.content}")
    
    print("=== [End of Memory Dump] ===\n")

def supervisor_router(state):
    if "sql_query" not in state:
        return "sql_agent"
    if "viz_path" not in state:
        return "viz_agent"
    return "final_summary"


graph = StateGraph(AppState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("sql_agent", sql_node)
graph.add_node("viz_agent", viz_node)
graph.add_node("final_summary", final_summary_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", supervisor_router, {
    "sql_agent": "sql_agent",
    "viz_agent": "viz_agent",
    "final_summary": "final_summary"
})
graph.add_edge("sql_agent", "supervisor")

graph.add_node("sql_checker", sql_checker_node)

# Route from sql_agent → either to supervisor or sql_checker (if error)
def sql_router(state):
    if "sql_error" in state and state["sql_error"]:
        return "sql_checker"
    return "supervisor"

graph.add_conditional_edges("sql_agent", sql_router, {
    "sql_checker": "sql_checker",
    "supervisor": "supervisor"
})

# Link sql_checker back to sql_agent (for retry)
graph.add_edge("sql_checker", "sql_agent")

graph.add_edge("viz_agent", "supervisor")
graph.add_edge("final_summary", END)

app_graph = graph.compile()

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         question = request.form['question']
#         state = app_graph.invoke({"question": question}, config={"recursion_limit": 10})
#         return render_template('results.html', results={
#             "question": state["question"],
#             "answer": state.get("answer", ""),
#             "sql_query": state.get("sql_query", ""),
#             "visualization": state.get("viz_path"),
#             "data": state.get("df", pd.DataFrame()).to_dict("records")[:10]
#         })
#     return render_template('index.html')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         question = request.form['question']
#         state = app_graph.invoke({"question": question}, config={"recursion_limit": 10})

#         # Print conversational history
#         print("\n=== [Conversation History] ===")
#         for i, msg in enumerate(memory.chat_memory.messages):
#             role = "User" if msg.type == "human" else "AI"
#             print(f"[{role}] {msg.content}")
#         print("===============================\n")

#         # Print final complete state
#         print("\n=== [FINAL STATE] ===")
#         print(state)
#         print("=====================\n")

#         return render_template('results.html', results={
#             "question": state["question"],
#             "answer": state.get("answer", ""),
#             "sql_query": state.get("sql_query", ""),
#             "visualization": state.get("viz_path"),
#             "data": state.get("df", pd.DataFrame()).to_dict("records")[:10]
#         })
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        state = app_graph.invoke({"question": question}, config={"recursion_limit": 10})

        response_data = {
            "question": state["question"],
            "answer": state.get("answer", ""),
            "sql_query": state.get("sql_query", ""),
            "visualization": state.get("viz_path"),
            "data": state.get("df", pd.DataFrame()).to_dict("records")[:10]
        }
        return response_data  # Return JSON response!

    return render_template('chat.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

