from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import json
import urllib.parse
import pandas as pd
import uuid
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
import socket
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from flask import send_from_directory

from agent.final_supervisor_agent_report import create_orchestrator

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-production')

# Keep the folder around for compatibility (older builds stored PNGs here),
# but visualization is now rendered client-side with Plotly.
os.makedirs(os.path.join(app.root_path, 'static', 'images'), exist_ok=True)

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

def _normalize_postgres_url(db_url: str) -> str:
    db_url = (db_url or "").strip()
    if db_url.startswith("postgres://"):
        return "postgresql://" + db_url[len("postgres://") :]
    return db_url


def _get_allowed_db_host_suffixes() -> tuple[str, ...]:
    """Return allowed hostname substrings for user-provided db_url.

    Defaults to common managed Postgres providers. Override with ALLOWED_DB_HOST_SUFFIXES.

    Note: despite the name, values are treated as *substrings* (to support pooler hosts like
    aws-...pooler.supabase.com).
    """

    raw = (os.environ.get("ALLOWED_DB_HOST_SUFFIXES") or "").strip()
    if raw:
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        return tuple(parts)
    return ("supabase", "neon")


def _is_allowed_db_host(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    host = hostname.lower().strip().rstrip(".")
    # Allow local dev (Docker compose: db) explicitly.
    if host in {"localhost", "127.0.0.1", "::1", "db"}:
        return True
    allowed_substrings = _get_allowed_db_host_suffixes()
    return any(s in host for s in allowed_substrings)


def _resolve_ipv4(hostname: str) -> Optional[str]:
    """Resolve an IPv4 address for hostname (returns first A record if any)."""
    try:
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not infos:
            return None
        sockaddr = infos[0][4]
        return str(sockaddr[0])
    except Exception:
        return None


def _with_hostaddr_ipv4(db_url: str, ipv4: str) -> str:
    """Add libpq hostaddr=IPv4 to force IPv4 routing when IPv6 is unavailable."""
    parsed = urllib.parse.urlparse(db_url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    qs["hostaddr"] = [ipv4]
    new_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def _apply_password_to_db_url(db_url: str, password: str) -> str:
    """If db_url has a username but no password, inject the provided password."""
    if not password:
        return db_url
    parsed = urllib.parse.urlparse(db_url)
    if not parsed.username:
        return db_url
    if parsed.password:
        return db_url

    user = urllib.parse.quote(parsed.username, safe="")
    pwd = urllib.parse.quote(password, safe="")
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{user}:{pwd}@{host}{port}"
    return urllib.parse.urlunparse(
        (parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _ensure_sslmode_require(db_url: str) -> str:
    parsed = urllib.parse.urlparse(db_url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    if "sslmode" in {k.lower() for k in qs.keys()}:
        return db_url
    qs["sslmode"] = ["require"]
    new_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def _with_search_path(db_url: str, schema_name: str) -> str:
    """Return db_url with Postgres search_path pinned to schema_name."""
    parsed = urllib.parse.urlparse(db_url)
    qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    # libpq/psycopg2: options='-c search_path=...'
    qs["options"] = [f"-c search_path={schema_name}"]
    new_query = urllib.parse.urlencode(qs, doseq=True)
    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def _validate_db_url(db_url: str) -> None:
    """Connectivity check; raises on failure."""
    def _try(url: str) -> None:
        engine = sqlalchemy.create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    try:
        _try(db_url)
        return
    except OperationalError as e:
        msg = str(e).lower()
        # Common on hosts without IPv6 egress (e.g., Render): libpq tries AAAA first.
        if "network is unreachable" not in msg and "no route to host" not in msg:
            raise

        parsed = urllib.parse.urlparse(db_url)
        if not parsed.hostname:
            raise
        ipv4 = _resolve_ipv4(parsed.hostname)
        if not ipv4:
            raise
        _try(_with_hostaddr_ipv4(db_url, ipv4))

def exec_code(python_code: str) -> Any:
    """Executes the provided Python code and returns a Plotly figure JSON spec.

    This intentionally avoids server-side image export (Kaleido/Chromium), which can OOM
    on small instances (e.g., Render free tier). The browser renders the interactive chart.
    """
    local_vars: dict[str, Any] = {}
    exec(python_code, {}, local_vars)
    fig_object = local_vars.get('fig', None)

    if not fig_object:
        print("No figure object found in the executed code.")
        return None

    try:
        # to_json() produces a fully JSON-serializable string.
        return json.loads(fig_object.to_json())
    except Exception as e:
        print(f"Visualization serialization failed: {str(e)}")
        return None


@app.route('/upload', methods=['POST'])
def upload_data():
    """Handle database connection + OpenAI API key upload."""
    try:
        # Get form data
        openai_api_key = request.form.get('openai_api_key', '').strip()
        db_url = request.form.get('db_url', '').strip()
        db_password = request.form.get('db_password', '').strip()
        schema_name = request.form.get('schema_name', '').strip() or None
        
        if not openai_api_key:
            return jsonify({'error': 'Please provide an OpenAI API key'}), 400

        if not db_url:
            return jsonify({'error': 'Please provide a PostgreSQL connection string (db_url)'}), 400

        db_url = _normalize_postgres_url(db_url)
        parsed = urllib.parse.urlparse(db_url)
        if parsed.scheme not in ("postgresql", "postgres"):
            return jsonify({'error': 'Only PostgreSQL connection strings are supported (postgresql://...)'}), 400

        if not _is_allowed_db_host(parsed.hostname):
            return jsonify(
                {
                    'error': (
                        'Database host is not allowed / Special characters in URL - please consider putting password below or encoding the url special chars. Use Supabase/Neon (or set ALLOWED_DB_HOST_SUFFIXES). '
                        f'Got host: {parsed.hostname}'
                    )
                }
            ), 400

        # If user provided password separately, inject it when the URL has no password.
        if db_password:
            db_url = _apply_password_to_db_url(db_url, db_password)

        if schema_name:
            db_url = _with_search_path(db_url, schema_name)
        
        # Create session ID for this user
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Validate connectivity (try as-is, then retry with sslmode=require if needed)
        try:
            _validate_db_url(db_url)
        except Exception:
            try:
                db_url_ssl = _ensure_sslmode_require(db_url)
                _validate_db_url(db_url_ssl)
                db_url = db_url_ssl
            except Exception as e:
                return jsonify(
                    {
                        'error': (
                            'Could not connect to the provided database URL. '
                            'If this is Supabase/Neon, ensure the user/password are correct and SSL is enabled '
                            '(try adding ?sslmode=require). '
                            f'Details: {str(e)}'
                        )
                    }
                ), 400

        # Create custom orchestrator with user's API key + DB URL
        user_orchestrator = create_orchestrator(openai_api_key, db_url=db_url, schema=schema_name)
        
        # Store user data in memory
        user_data_store[session_id] = {
            'openai_api_key': openai_api_key,
            'db_url': db_url,
            'schema_name': schema_name,
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
                'error': 'Please connect your database and API key first'
            }), 400
        
        question = request.form['question']
        state = user_data_store[session_id].get('state') or _default_agent_state()
        state["messages"].append(HumanMessage(content=question))
        print("LATEST MESSAGE IS:", state["messages"][-1].content)

        # Use user's custom orchestrator
        user_orchestrator = user_data_store[session_id]['orchestrator']
        state = user_orchestrator.invoke(state)
        plotly_figure = exec_code(state["python_visualization_code"])
        if plotly_figure:
            # Keep report history, but omit server-side images (client renders interactively).
            state["report_states"].append({
                "question": state["question"],
                "img_path": "",
                "summary": state["messages"][-1].content,
            })

        response_data = {
            "question": state["question"],
            "answer": state["messages"][-1].content,
            "sql_query": state["sql_query"],
            "visualization": None,
            "plotly_figure": plotly_figure,
            "data": state["df"].to_dict(orient="records") if isinstance(state["df"], pd.DataFrame) else state["df"],
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


@app.route('/reset', methods=['POST'])
def reset_session():
    """Clear the current session's in-memory state so the user can re-initialize."""
    try:
        session_id = session.get('session_id')
        if session_id:
            user_data_store.pop(session_id, None)
        session.pop('session_id', None)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Failed to reset session: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(port=3000, debug=False, use_reloader=False)

