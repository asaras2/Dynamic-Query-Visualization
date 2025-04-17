
from flask import Flask, render_template, request
import os, urllib.parse, re, pandas as pd, sqlalchemy, uuid, requests
import plotly.graph_objects as go
import plotly.express as px
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Azure LLaMA configuration
AZURE_API_ENDPOINT = "https://Meta-Llama-3-1-70B-Instruct-zhlk.westus.models.ai.azure.com/chat/completions"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

print("Azure Key Loaded:", bool(AZURE_API_KEY))

def call_llama(messages, max_tokens=1024, temperature=0.3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(AZURE_API_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    else:
        raise Exception(f"Azure LLaMA API Error {response.status_code}: {response.text}")

def setup_database_connection():
    driver = '{ODBC Driver 18 for SQL Server}'
    SERVER_NAME = "th1.database.windows.net"
    DATABASE_NAME = "Th-1-pgsql"
    USERNAME = "th1-admin"
    PASSWORD = "ThreSh0lD@01"
    odbc_str = (
        f"DRIVER={driver};SERVER={SERVER_NAME},1433;"
        f"DATABASE={DATABASE_NAME};UID={USERNAME};PWD={PASSWORD};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    connect_str = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)
    engine = sqlalchemy.create_engine(connect_str)
    return SQLDatabase(engine, schema="SalesLT")

db = setup_database_connection()

def generate_visualization(df, query, question):
    df_description = f"""User Question: {question}\nSQL Query: {query}\nColumns: {', '.join(df.columns)}\nPreview:\n{df.head(3)}"""
    prompt = [
        {"role": "user", "content": f"You are a data visualization expert. Based on this: {df_description}\nSuggest a chart type, x/y axes, and title."}
    ]
    recommendation = call_llama(prompt)
    rec_dict = {}
    for line in recommendation.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            rec_dict[key.strip()] = value.strip()

    plotly_prompt = [
        {"role": "user", "content": f"Generate Plotly code in Python for: {recommendation}\nUse df in scope. Output only the code."}
    ]
    plotly_code = call_llama(plotly_prompt)
    plotly_code = re.sub(r'```.*?```', '', plotly_code, flags=re.DOTALL).strip()

    local_vars = {'df': df, 'go': go, 'px': px, 'pd': pd}
    exec(plotly_code, {}, local_vars)
    fig = local_vars.get("fig")
    if fig:
        filename = f"viz_{uuid.uuid4().hex}.png"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fig.write_image(path)
        return filename
    return None

def process_user_question(user_question):
    try:
        table_info = db.get_table_info()
        prompt = [{"role": "user", "content": f"Generate a SQL query for: {user_question}\nAvailable tables:\n{table_info}"}]
        sql_query = call_llama(prompt)
        sql_query = re.sub(r'```sql|```', '', sql_query).strip()

        query_tool = QuerySQLDatabaseTool(db=db)
        query_result = query_tool.invoke(sql_query)
        df = pd.read_sql(sql_query, db._engine)
        viz_filename = generate_visualization(df, sql_query, user_question) if not df.empty else None

        answer_prompt = [{"role": "user", "content": f"Explain in simple language: {user_question} using this data: {query_result}"}]
        answer = call_llama(answer_prompt)

        return {
            "question": user_question,
            "answer": answer,
            "sql_query": sql_query,
            "visualization": viz_filename,
            "data": df.to_dict('records')[:10]
        }
    except Exception as e:
        return {
            "question": user_question,
            "answer": f"Error: {str(e)}",
            "sql_query": "",
            "visualization": None,
            "data": []
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        results = process_user_question(question)
        return render_template('results.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
