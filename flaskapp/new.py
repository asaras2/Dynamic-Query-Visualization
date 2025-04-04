import os
import re
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
# from langgraph.pregel import GraphRecorder
from langgraph.graph.graph import Graph
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from graphviz import Digraph
from openai import OpenAI
import sqlalchemy
import urllib

# ======================
# CONNECTION SETTINGS
# ======================
def setup_database_connection():
    """Establish database connection with proper error handling"""
    driver = '{ODBC Driver 18 for SQL Server}'
    SERVER_NAME = "th1.database.windows.net"
    DATABASE_NAME = "Th-1-pgsql"
    USERNAME = "th1-admin"
    PASSWORD = "ThreSh0lD@01"  # In production, use environment variables
    
    if not PASSWORD:
        raise ValueError("Database password not found in environment variables")
    
    try:
        odbc_str = (
            f"DRIVER={driver};"
            f"SERVER={SERVER_NAME},1433;"
            f"DATABASE={DATABASE_NAME};"
            f"UID={USERNAME};"
            f"PWD={PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
        )
        connect_str = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str)
        engine = sqlalchemy.create_engine(connect_str)
        db = SQLDatabase(engine, schema="SalesLT")
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to establish database connection: {str(e)}")

# Initialize database connection
db = setup_database_connection()

# Initialize NVIDIA LLM client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

# ======================
# STATE DEFINITION
# ======================
class AgentState(TypedDict):
    """State of our agentic workflow"""
    question: str
    sql_query: Optional[str]
    db_result: Optional[str]
    intermediate_answer: Optional[str]
    final_answer: Optional[str]
    visualization: Optional[str]
    should_visualize: Optional[bool] 

# ======================
# TOOLS & AGENTS
# ======================
def generate_sql_query(question: str, table_info: str) -> str:
    """Generate SQL query using NVIDIA LLM (strict output format)"""
    prompt = f"""
    You are a SQL expert. Given the following database schema and question, 
    generate ONLY the SQL query to answer the question. 
    
    RULES:
    1. Return ONLY the SQL query text
    2. No markdown formatting (no ```sql ```)
    3. No explanations or additional text
    4. Include the semicolon at the end
    
    Database Schema:
    {table_info}
    
    Question: {question}
    
    SQL Query:"""
    
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        top_p=0.7,
        max_tokens=1024
    )
    
    # Extract just the pure SQL query
    raw_query = response.choices[0].message.content.strip()
    
    # Clean any accidental markdown formatting
    query = re.sub(r'```sql|```', '', raw_query).strip()
    
    # Ensure semicolon at end
    if not query.endswith(';'):
        query += ';'
        
    print("Generated SQL:", query)  # Debug print
    return query

def execute_sql_query(query: str) -> str:
    """Execute SQL query against the database"""
    execute_query = QuerySQLDataBaseTool(db=db)
    try:
        result = execute_query.run(query)
        print("Query Execution Result:", result)

        # Convert to list of tuples if needed
        if isinstance(result, str):
            try:
                import ast
                return ast.literal_eval(result)
            except:
                return result
        return result
    
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(error_msg)
        return error_msg

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.io import to_html
import io
import base64

def generate_visualization(result: str, query: str, question: str) -> Optional[dict]:
    """
    Generate visualization using LLM-generated Matplotlib code
    Returns: {'image': base64_string, 'type': 'matplotlib'}
    """
    try:
        # Convert result to DataFrame
        if isinstance(result, str):
            try:
                import ast
                data = ast.literal_eval(result)
                df = pd.DataFrame(data)
            except (ValueError, SyntaxError):
                try:
                    data = json.loads(result)
                    df = pd.DataFrame(data)
                except json.JSONDecodeError:
                    return None
        else:
            df = pd.DataFrame(result)

        if df.empty:
            return None

        # Generate visualization prompt with strict requirements
        prompt = f"""Generate professional-quality Python code for data visualization 
                    using Matplotlib based on the following data and question. Act as a visualization 
                    specialist who prioritizes clarity, aesthetics, and effective communication.
        
        Data:
        {df.head().to_string()}
        
        Question: {question}

        Role: suggest an appropriate graph according to answer: {result}, query: {query}, and  Question: {question}, then give the code for that graph visualization using matplotlib
        and follow the requirements and syntax given below.
        
        Requirements:
        1. Use matplotlib.pyplot as plt
        2. Choose the most appropriate chart type (bar, line, pie, etc.)
        3. Include proper title, labels, and legend if needed
        4. Use figsize=(10,6)
        5. Rotate x-axis labels if needed (plt.xticks(rotation=45))
        6. Include plt.tight_layout()
        7. Save to BytesIO buffer and return it
        8. Close the figure after saving
        9. generate lengends for these graphs, and use the data visualization regeims for making the graphs By keeping
        
        Return ONLY the code in this exact format:
        ```python
        plt.figure(figsize=(10, 6))
        # Visualization code here
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        return buf
        ```
        """

        # Get visualization code from LLM
        response = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )

        plotly_code = response.choices[0].message.content.strip()
        print("Generated Visualization Code:\n", plotly_code)
        
        # Extract code from markdown block
        if '```python' in plotly_code:
            plotly_code = plotly_code.split('```python')[1].split('```')[0]
        elif '```' in plotly_code:
            plotly_code = plotly_code.split('```')[1].split('```')[0]

        # Prepare execution environment
        local_vars = {
            'df': df,
            'plt': plt,
            'BytesIO': BytesIO,
            'pd': pd
        }

        # Execute the generated code
        exec(plotly_code, globals(), local_vars)
        buf = local_vars.get('buf')

        if buf and isinstance(buf, BytesIO):
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return {
                'image': image_base64,
                'type': 'matplotlib',
                'question': question,
                'query': query
            }
        
        # # Fallback if LLM-generated code fails
        # print("Falling back to default visualization")
        # plt.figure(figsize=(10, 6))
        # if len(df.columns) >= 2:
        #     df.plot(kind='bar' if len(df) < 10 else 'line', 
        #            x=df.columns[0], 
        #            legend=len(df.columns) > 2)
        # else:
        #     plt.hist(df.iloc[:, 0], bins=10)
        # plt.title(question)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        
        # buf = BytesIO()
        # plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        # plt.close()
        
        # buf.seek(0)
        # image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # return {
        #     'image': image_base64,
        #     'type': 'matplotlib',
        #     'question': question,
        #     'query': query
        # }

    except Exception as e:
        print(f"Visualization generation failed: {str(e)}")
        return None
    
def generate_natural_language_answer(question: str, query: str, result: str) -> str:
    """Generate natural language answer from SQL results"""
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, 
        answer the user question in natural language. Be concise but informative.
        
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        
        Answer: """
    )
    
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{
            "role": "user",
            "content": answer_prompt.format(
                question=question,
                query=query,
                result=result,
            )
        }],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    
    answer = response.choices[0].message.content.strip()
    print("Generated Answer:", answer)
    return answer

# ======================
# NODE FUNCTIONS
# ======================
def sql_agent(state: AgentState) -> dict:
    """Generate and execute SQL query"""
    question = state["question"]
    table_info = db.get_table_info()
    
    try:
        sql_query = generate_sql_query(question, table_info)
        db_result = execute_sql_query(sql_query)
        
        return {
            "sql_query": sql_query,
            "db_result": db_result
        }
    except Exception as e:
        return {
            "sql_query": None,
            "db_result": f"Error in SQL generation/execution: {str(e)}"
        }
    

def answer_agent(state: AgentState) -> dict:
    """Generate natural language answer from SQL results"""
    question = state["question"]
    sql_query = state["sql_query"]
    db_result = state["db_result"]
    
    if not sql_query or not db_result:
        return {
            "intermediate_answer": "Failed to generate SQL query or execute it.",
            "should_visualize": False
        }
    
    answer = generate_natural_language_answer(question, sql_query, db_result)
    
    # Determine if visualization is appropriate
    should_visualize = True
    # try:
    #     # Check if results are visualizable
    #     if isinstance(db_result, str):
    #         if "error" in db_result.lower() or "no results" in db_result.lower():
    #             should_visualize = False
    #     elif len(db_result) == 0:  # Empty results
    #         should_visualize = False
        
    #     # Additional check for visualization compatibility
    #     if should_visualize:
    #         # Try to create a DataFrame to verify if visualization is possible
    #         test_df = pd.DataFrame(db_result if not isinstance(db_result, str) else [])
    #         if test_df.empty or len(test_df.columns) < 1:
    #             should_visualize = False
                
    # except Exception as e:
    #     print(f"Visualization check failed: {str(e)}")
    #     should_visualize = False
    
    return {
        "intermediate_answer": answer,
        "should_visualize": should_visualize,
        "sql_query": sql_query,
        "db_result": db_result
    }

def visualization_agent(state: AgentState) -> dict:
    try:
        db_result = state["db_result"]
        print("Raw DB result:", db_result)  # Debug
        
        # Convert string results to Python objects
        if isinstance(db_result, str):
            try:
                import ast
                db_result = ast.literal_eval(db_result)
            except (ValueError, SyntaxError):
                try:
                    db_result = json.loads(db_result)
                except json.JSONDecodeError:
                    return {"visualization": None}
        
        # Ensure we have visualizable data
        if not db_result or len(db_result) == 0:
            return {"visualization": None}
            
        visualization = generate_visualization(
            db_result,
            state["sql_query"],
            state["question"]
        )
        
        # Return the visualization object directly
        return {
            "visualization": visualization['image'] if visualization else None,  # Just the base64 string
            "intermediate_answer": state["intermediate_answer"],
            "sql_query": state["sql_query"],
            "db_result": state["db_result"]
        }
        
    except Exception as e:
        print(f"Visualization agent error: {str(e)}")
        return {"visualization": None}

def output_formatter(state: AgentState) -> dict:
    """Format final output with all necessary fields"""
    return {
        "final_answer": state.get("intermediate_answer", "No answer generated"),
        "sql_query": state.get("sql_query", ""),
        "db_result": state.get("db_result"),
        "visualization": state.get("visualization")  # This is now just the base64 string
    }

# ======================
# GRAPH DEFINITION
# ======================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("sql_agent", sql_agent)
workflow.add_node("answer_agent", answer_agent)
workflow.add_node("visualization_agent", visualization_agent)  # Add visualization node
workflow.add_node("output_formatter", output_formatter)

# Set entry point
workflow.set_entry_point("sql_agent")

# Add edges
workflow.add_edge("sql_agent", "answer_agent")
workflow.add_edge("answer_agent", "visualization_agent")  # Always go to visualization
workflow.add_edge("visualization_agent", "output_formatter")
workflow.add_edge("output_formatter", END)

# Compile the workflow
app = workflow.compile()

# # ======================
# # GRAPH DEFINITION
# # ======================
# workflow = StateGraph(AgentState)

# # Add nodes
# workflow.add_node("sql_agent", sql_agent)
# workflow.add_node("answer_agent", answer_agent)
# workflow.add_node("output_formatter", output_formatter)

# # Set entry point
# workflow.set_entry_point("sql_agent")

# # Add edges
# workflow.add_edge("sql_agent", "answer_agent")
# workflow.add_edge("answer_agent", "output_formatter")
# workflow.add_edge("output_formatter", END)

# # save_langgraph_visualization(workflow)

# # Compile the workflow
# app = workflow.compile()


# ======================
# EXECUTION FUNCTION
# ======================

def run_agentic_pipeline(question: str) -> dict:
    try:
        initial_state = {
            "question": question,
            "retry_count": 0,
            "should_visualize": True
        }
        
        result = app.invoke(initial_state)

        # Build the complete response
        response = {
            "final_answer": result.get("final_answer", "No answer generated"),
            "sql_query": result.get("sql_query", ""),
            "status": "success",
            "error": None,
            "db_result": result.get("db_result"),
            "visualization": result.get("visualization")  # This is the base64 string
        }

        print("DEBUG RESPONSE:", {**response, "visualization": "BASE64_IMAGE" if response["visualization"] else None})  # Don't print full base64
        return response

    except Exception as e:
        print("DEBUG ERROR:", str(e))
        return {
            "final_answer": "Error generating answer",
            "sql_query": "",
            "status": "error",
            "error": str(e),
            "db_result": None,
            "visualization": None
        }

# def run_agentic_pipeline(question: str) -> dict:
#     try:
#         initial_state = {"question": question, "retry_count": 0}
#         result = app.invoke(initial_state)

#         # Get the natural language answer from the agent
#         final_answer = result.get("final_answer", "No answer generated")
        
#         # Still include the query for reference
#         sql_query = result.get("sql_query", "")
        
#         # Include the db_result only if needed for debugging
#         db_result = result.get("db_result", None)

#         response = {
#             "answer": str(final_answer),  # The natural language response
#             "query": sql_query,          # The generated SQL query
#             "status": result.get("security_status", "unknown"),
#             "error": result.get("error"),
#             "db_result": db_result       # Optional: include for debugging
#         }

#         print("DEBUG RESPONSE:", response)
#         return response

#     except Exception as e:
#         print("DEBUG ERROR:", str(e))
#         return {
#             "answer": "Error generating answer",
#             "query": "",
#             "status": "system_error",
#             "error": str(e)
#         }

