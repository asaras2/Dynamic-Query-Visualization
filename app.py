from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from dotenv import load_dotenv
import os
import urllib.parse
import pandas as pd
import re
import uuid
import sqlalchemy
import tempfile
import subprocess
import shutil
import psycopg2
from psycopg2 import sql as pg_sql
from werkzeug.utils import secure_filename
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
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

from agent.final_supervisor_agent_report import create_orchestrator

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['SQL_UPLOAD_FOLDER'] = 'uploads/sql'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SQL_UPLOAD_FOLDER'], exist_ok=True)

# In-memory storage for uploaded data (session-based)
# In production, consider using Redis or database
user_data_store = {}

def _default_agent_state():
    return {
        "messages": [],
        "question": "",
        "sql_query": "",
        "results": "",
        "df": "",
        "python_visualization_code": "",
        "report_states": [],
    }


def _get_pg_admin_conninfo() -> dict:
    """Return connection info for the *admin* database used to CREATE DATABASE."""
    # Prefer a single URL in env.
    admin_url = os.environ.get("POSTGRES_ADMIN_URL") or os.environ.get("PG_ADMIN_URL")
    if admin_url:
        parsed = urllib.parse.urlparse(admin_url)
        if parsed.scheme not in ("postgres", "postgresql"):
            raise ValueError("POSTGRES_ADMIN_URL must start with postgresql://")
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": urllib.parse.unquote(parsed.username or "postgres"),
            "password": urllib.parse.unquote(parsed.password or ""),
            "dbname": (parsed.path.lstrip("/") or "postgres"),
        }

    # Fallback to discrete PG env vars.
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
        "dbname": os.environ.get("PGDATABASE", "postgres"),
    }


def _build_pg_url(conninfo: dict, *, dbname: str) -> str:
    user = urllib.parse.quote(conninfo["user"])
    password = urllib.parse.quote(conninfo.get("password", ""))
    host = conninfo["host"]
    port = conninfo["port"]
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return f"postgresql://{user}@{host}:{port}/{dbname}"


def _create_database_if_missing(conninfo: dict, *, dbname: str) -> None:
    """Create a Postgres database if it doesn't exist."""
    conn = psycopg2.connect(
        host=conninfo["host"],
        port=conninfo["port"],
        user=conninfo["user"],
        password=conninfo.get("password", ""),
        dbname=conninfo["dbname"],
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone():
                return
            cur.execute(pg_sql.SQL("CREATE DATABASE {}" ).format(pg_sql.Identifier(dbname)))
    finally:
        conn.close()


def _import_sql(db_url: str, sql_file_path: str, *, conninfo: dict) -> None:
    """Import a .sql file into the given db_url.

    Prefers `psql` when available (handles most dumps). Falls back to a simple psycopg2 executor
    which will NOT work for complex dumps (e.g., COPY ... FROM STDIN).
    """
    psql = shutil.which("psql")
    if psql:
        env = os.environ.copy()
        if conninfo.get("password"):
            env["PGPASSWORD"] = conninfo["password"]

        proc = subprocess.run(
            [psql, db_url, "-v", "ON_ERROR_STOP=1", "-f", sql_file_path],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(f"psql import failed: {stderr}")
        return
    else:
        # error
        raise RuntimeError("psql command not found. Please ensure psql is installed and in your PATH for SQL imports.")

    # # Fallback: basic statement splitting for simple schema+inserts.
    # with open(sql_file_path, "r", encoding="utf-8", errors="ignore") as f:
    #     sql_text = f.read()

    # conn = psycopg2.connect(db_url)
    # try:
    #     with conn.cursor() as cur:
    #         statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    #         for stmt in statements:
    #             cur.execute(stmt)
    #     conn.commit()
    # finally:
    #     conn.close()

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


@app.route('/upload', methods=['POST'])
def upload_data():
    """Handle SQL file and OpenAI API key upload."""
    try:
        # Get uploaded files and form data
        sql_file = request.files.get('sql_file')
        openai_api_key = request.form.get('openai_api_key', '').strip()
        
        if not sql_file or not sql_file.filename:
            return jsonify({'error': 'Please upload a SQL file'}), 400
            
        if not openai_api_key:
            return jsonify({'error': 'Please provide an OpenAI API key'}), 400
            
        if not sql_file.filename.lower().endswith('.sql'):
            return jsonify({'error': 'Please upload a .sql file'}), 400
        
        # Create session ID for this user
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Save SQL file to local storage
        filename = secure_filename(sql_file.filename)
        sql_file_path = os.path.join(app.config['SQL_UPLOAD_FOLDER'], f"{session_id}_{filename}")
        sql_file.save(sql_file_path)
        
        # Create a dedicated Postgres database for this upload and import the SQL into it
        try:
            conninfo = _get_pg_admin_conninfo()
            safe_id = session_id.replace("-", "")
            db_name = f"dqv_{safe_id}"[:63]
            _create_database_if_missing(conninfo, dbname=db_name)
            db_url = _build_pg_url(conninfo, dbname=db_name)
            _import_sql(db_url, sql_file_path, conninfo=conninfo)
        except Exception as e:
            return jsonify(
                {
                    "error": (
                        "Failed to initialize Postgres database from uploaded SQL. "
                        "Ensure a Postgres server is running and POSTGRES_ADMIN_URL (or PGHOST/PGUSER/PGPASSWORD/...) is set. "
                        f"Details: {str(e)}"
                    )
                }
            ), 500

        # Create custom orchestrator with user's API key + DB URL
        user_orchestrator = create_orchestrator(openai_api_key, db_url=db_url)
        
        # Store user data in memory
        user_data_store[session_id] = {
            'sql_file_path': sql_file_path,
            'openai_api_key': openai_api_key,
            'db_name': db_name,
            'db_url': db_url,
            'orchestrator': user_orchestrator,
            'state': _default_agent_state(),
            'uploaded_at': pd.Timestamp.now()
        }
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded successfully',
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if user has uploaded their data
        session_id = session.get('session_id')
        if not session_id or session_id not in user_data_store:
            return jsonify({
                'error': 'Please upload your SQL file and API key first'
            }), 400
        
        question = request.form['question']
        state = user_data_store[session_id].get('state') or _default_agent_state()
        state["messages"].append(HumanMessage(content=question))
        print("LATEST MESSAGE IS:", state["messages"][-1].content)

        # Use user's custom orchestrator
        user_orchestrator = user_data_store[session_id]['orchestrator']
        state = user_orchestrator.invoke(state)
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
        
        user_data_store[session_id]['state'] = state

        print("Response Data:", response_data)

        return response_data  # Return JSON response!

    session_id = session.get('session_id')
    is_initialized = bool(session_id and session_id in user_data_store)
    return render_template('chat.html', is_initialized=is_initialized)

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

