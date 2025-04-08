from flask import Flask, render_template, request, jsonify
from textTosql import run_agentic_pipeline
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, resources={
    r"/query": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get("question", "")
        print(f"Received question: {question}")

        response = run_agentic_pipeline(question)
        # print("Pipeline response:", {**response, "visualization": "BASE64_IMAGE" if response.get("visualization") else None})

        return jsonify({
            "answer": response.get("final_answer", "No answer generated"),
            "query": response.get("sql_query", ""),
            "visualization": response.get("visualization"),  # Base64 string
            "status": response.get("status", "success"),
            "error": response.get("error"),
            "db_result": response.get("db_result")
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "answer": "Error processing request",
            "status": "error",
            "error": str(e),
            "query": "",
            "visualization": None,
            "db_result": None
        }), 500
    
if __name__ == "__main__":
    app.run(debug=True)