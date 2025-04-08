from flask import Flask, render_template, request, redirect, url_for
import os
import urllib.parse
import re
import pandas as pd
import sqlalchemy
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
import uuid

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Initialize OpenAI Client
client = OpenAI(
    api_key= "api_key"  # Make sure to set this environment variable
)  # This will use the OPENAI_API_KEY environment variable

# YOUR EXACT generate_visualization FUNCTION HERE
def generate_visualization(df, query, question):
    """
    Generate an appropriate visualization for the query results using OpenAI GPT-4.
    
    Args:
        df (pd.DataFrame): The query results
        query (str): The SQL query
        question (str): The original user question
        
    Returns:
        str: Path to the saved image file
    """
    # Create a text description of the dataframe
    df_description = f"""
    User Question: {question}
    
    SQL Query: {query}
    
    Resulting Data:
    - Number of rows: {len(df)}
    - Number of columns: {len(df.columns)}
    - Columns: {', '.join(df.columns)}
    - First 3 rows:
    {df.head(3).to_string()}
    
    Data Types:
    {df.dtypes.to_string()}
    """
    
    # Prompt for visualization recommendation
    prompt = f"""You are a data visualization expert. Analyze the following SQL query results and recommend the most appropriate visualization.

    {df_description}

    Respond with:
    1. The recommended visualization type (bar, line, pie, scatter, histogram, etc.)
    2. The columns to use for x-axis, y-axis, etc.
    3. A suggested title
    4. A brief explanation of why this visualization fits

    Format your response as:
    Visualization Type: <type>
    X-Axis: <column or none>
    Y-Axis: <column or none>
    Color: <column or none>
    Title: <suggested title>
    Explanation: <brief explanation>"""
    
    # Get visualization recommendation
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    
    recommendation = response.choices[0].message.content.strip()
    print("\nVisualization Recommendation:\n", recommendation)
    
    # Parse the recommendation
    rec_dict = {}
    for line in recommendation.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            rec_dict[key.strip()] = value.strip()
    
    # Generate Plotly code prompt - ask for clean code without markdown
    plotly_prompt = f"""Create Plotly visualization code in Python based on these specifications:

    Data Sample:
    {df.head(3).to_string()}

    Visualization Specifications:
    {recommendation}

    Generate complete Python code using Plotly that:
    1. Creates the visualization
    2. Includes proper axis labels and title
    3. Handles any necessary data transformations
    4. Returns the fig object

    The code should:
    - Work with a pandas DataFrame called 'df' in scope
    - Use either plotly.graph_objects (go) or plotly.express (px)
    - Be properly indented
    - NOT include any markdown formatting or code blocks
    - NOT include any explanations or comments
    """
    
    # Get Plotly code
    plotly_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": plotly_prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    
    plotly_code = plotly_response.choices[0].message.content.strip()
    print("\nGenerated Plotly Code:\n", plotly_code)
    
    # Clean the code by removing any markdown blocks
    if '```python' in plotly_code:
        plotly_code = plotly_code.split('```python')[1].split('```')[0]
    elif '```' in plotly_code:
        plotly_code = plotly_code.split('```')[1].split('```')[0]
    
    # Prepare the execution environment
    local_vars = {
        'df': df,
        'go': go,
        'px': px,
        'pd': pd
    }
    
    try:
        # Execute the code in a controlled environment
        exec(plotly_code, globals(), local_vars)
        fig = local_vars.get('fig')
        
        if fig is None:
            # Try to find the figure object by checking for Plotly figures
            for var in local_vars.values():
                if isinstance(var, go.Figure):
                    fig = var
                    break
            
        if fig is None:
            raise ValueError("Generated code did not produce a Plotly figure object")
        
        # Save the figure with unique filename
        filename = f"viz_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        fig.write_image(output_path)
        print(f"\nVisualization saved to {output_path}")
        return filename
        
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        print("Attempting to create a basic visualization as fallback...")
        
        try:
            # Fallback visualization
            if len(df.columns) == 1:
                # Single column - show histogram or bar chart
                col = df.columns[0]
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                else:
                    fig = px.bar(df[col].value_counts(), title=f"Count of {col}")
            elif len(df.columns) == 2:
                # Two columns - assume x and y
                x_col, y_col = df.columns[0], df.columns[1]
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                else:
                    fig = px.bar(df[y_col].value_counts(), title=f"Count of {y_col}")
            else:
                # Multiple columns - show first two numeric columns
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                   title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                else:
                    # Last resort - show first two columns
                    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], 
                                   title=f"{df.columns[1]} vs {df.columns[0]}")
            
            filename = f"viz_{uuid.uuid4().hex}.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            fig.write_image(output_path)
            print(f"Fallback visualization saved to {output_path}")
            return filename
            
        except Exception as fallback_error:
            print(f"Failed to create fallback visualization: {str(fallback_error)}")
            return None

def process_user_question(user_question):
    """Process a user question and return results"""
    try:
        # Generate SQL Query
        prompt = f"""Generate a SQL query to answer: {user_question}. 
        The database has these tables:\n\n{db.get_table_info()}
        Return only the SQL query, nothing else."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r'```sql|```', '', sql_query).strip()
        
        # Execute query
        execute_query = QuerySQLDatabaseTool(db=db)
        query_result = execute_query.invoke(sql_query)
        
        # Convert to DataFrame
        df = pd.read_sql(sql_query, db._engine)
        
        # Generate visualization using your exact function
        viz_filename = generate_visualization(df, sql_query, user_question) if not df.empty else None
        
        # Generate natural language answer
        answer_prompt = f"""Answer this question in simple language: {user_question}
        using this data: {query_result}"""
        
        answer_response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.2,
            max_tokens=500
        )
        
        answer = answer_response.choices[0].message.content.strip()
        
        return {
            "question": user_question,
            "answer": answer,
            "sql_query": sql_query,
            "visualization": viz_filename,
            "data": df.to_dict('records')[:10]  # First 10 rows for display
        }
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return {
            "question": user_question,
            "answer": f"Sorry, I couldn't process your question. Error: {str(e)}",
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