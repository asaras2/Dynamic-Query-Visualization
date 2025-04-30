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
import plotly.graph_objects as go
import plotly.express as px
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import kaleido
import plotly.io as pio
import imgkit

from flask import send_from_directory

from agent.final_supervisor_agent_report import SQL_ORCHESTRATOR

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

current_state = {
    "messages": [],
    "question": "",
    "sql_query": "",
    "results": "",
    "df": "",
    "python_visualization_code": "",
    "report_states": []
}

def exec_code(python_code: str) -> Any:
    """Executes the provided Python code and returns the result."""
    local_vars = {}
    exec(python_code, {}, local_vars)
    filename = f"viz_{uuid.uuid4().hex}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    fig_object = local_vars.get('fig', None)

    if fig_object:
        png_bytes = fig_object.to_image(format="png", width=1200, height=800)
        with open(output_path, "wb") as f:
            f.write(png_bytes)
    else:
        print("No figure object found in the executed code.")
    
    return filename


@app.route('/', methods=['GET', 'POST'])
def index():
    global current_state
    if request.method == 'POST':
        question = request.form['question']
        current_state["messages"].append(HumanMessage(content=question))
        print("LATEST MESSAGE IS:", current_state["messages"][-1].content)

        state = SQL_ORCHESTRATOR.invoke(current_state)
        filename = exec_code(state["python_visualization_code"])
        viz_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        state["report_states"].append({
            "question": state["question"],
            "img_path": viz_path,
            "summary": state["messages"][-1].content,
        })

        response_data = {
            "question": state["question"],
            "answer": state["messages"][-1].content,
            "sql_query": state["sql_query"],
            "visualization":filename,
            "data": state["df"].to_dict(orient="records")
        }
        
        current_state = state

        print("Response Data:", response_data)

        return response_data  # Return JSON response!

    return render_template('chat.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/reports', methods=['GET', 'POST'])
def reports():
    if request.method == 'POST':
        report_files = os.listdir("./reports")
        response_data = {
            "report_files": report_files
        }
        return response_data

    return render_template('reports.html')

@app.route('/reports/<filename>')
def download_report(filename):
    return send_from_directory('./reports', filename, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

