from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain_community.utilities import SQLDatabase
import sqlalchemy
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chains import create_sql_query_chain
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.tools import QuerySQLDatabaseTool
from operator import itemgetter
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain.tools import tool
from pydantic import BaseModel



import os
import urllib

# driver = '{ODBC Driver 17 for SQL Server}'
# server = os.environ["SERVER"]
# database = os.environ["DATABASE"]
# username = os.environ["USERNAME"]
# password = os.environ["PASSWORD"]

# odbc_str = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;UID='+username+';DATABASE='+ database + ';PWD='+ password
# connect_str = 'mssql+pyodbc:///?odbc_connect=' + urllib.parse.quote_plus(odbc_str)

# print("CONN STRING")
# print(connect_str)

# engine = sqlalchemy.create_engine(connect_str)
# db = SQLDatabase(engine, schema="SalesLT")

def setup_database_connection():
    user = "mehulmathur"
    psswd = "mehul160401"
    host = "localhost"
    port = "5432"
    db = "mehul_532"

    url = f"postgresql://{user}:{psswd}@{host}:{port}/{db}"
    engine = sqlalchemy.create_engine(url)
    try:
        db = SQLDatabase(engine, schema="analytical_schema")
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to establish database connection: {str(e)}")

# Initialize database connection
db = setup_database_connection()


import re

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.

    Args:
        text (str): Raw SQL query text that may contain code blocks, tags, and backticks

    Returns:
        str: Cleaned SQL query
    """
    # Step 1: Remove code block syntax and any SQL-related tags
    # This handles variations like ```sql, ```SQL, ```SQLQuery, etc.
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)

    # Step 2: Handle "SQLQuery:" prefix and similar variations
    # This will match patterns like "SQLQuery:", "SQL Query:", "MySQL:", etc.
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

    # Step 3: Extract the first SQL statement if there's random text after it
    # Look for a complete SQL statement ending with semicolon
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # Step 4: Remove backticks around identifiers
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Step 5: Normalize whitespace
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Step 6: Preserve newlines for main SQL keywords to maintain readability
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
               'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
               'OUTER JOIN', 'UNION', 'VALUES', 'INSERT', 'UPDATE', 'DELETE']

    # Case-insensitive replacement for keywords
    pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
    text = re.sub(f'({pattern})', r'\n\1', text, flags=re.IGNORECASE)

    # Step 7: Final cleanup
    # Remove leading/trailing whitespace and extra newlines
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text

class SQLToolSchema(BaseModel):
    question: str

llm = ChatOpenAI(model="gpt-4")

# @tool(args_schema=SQLToolSchema)
# def nl2sql_tool(question):
#     """Tool to Generate and Execute SQL Query to answer questions"""
#     print("INSIDE NL2SQL TOOL")
#     execute_query = QuerySQLDatabaseTool(db=db)
#     write_query = create_sql_query_chain(llm, db)

#     print("Query was ", write_query, " for question ", question)

#     chain = (
#         RunnablePassthrough.assign(query=write_query | RunnableLambda(clean_sql_query)).assign(
#             result=itemgetter("query") | execute_query
#         )
#     )

#     response = chain.invoke({"question": question})
#     print("Response was ", response)
#     return response['result']

# Helper function for executing cleaned SQL queries
def execute_sql_query(query: str) -> str:
    """Execute SQL query against the database."""
    execute_query = QuerySQLDatabaseTool(db=db)
    try:
        result = execute_query.run(query)
        print("Query Execution Result:", result)

        if isinstance(result, str):
            try:
                import ast
                return ast.literal_eval(result)
            except Exception:
                return result
        return result

    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        print(error_msg)
        return error_msg

# Tool definition
@tool(args_schema=SQLToolSchema)
def nl2sql_tool(question: str) -> str:
    """Tool to Generate and Execute SQL Query to answer questions."""
    print("INSIDE NL2SQL TOOL")

    # Step 1: SQL generation chain
    write_query = create_sql_query_chain(llm, db)

    # Step 2: Build the full runnable chain
    chain = (
        RunnablePassthrough.assign(
            query=write_query | RunnableLambda(clean_sql_query)
        ).assign(
            result=itemgetter("query") | RunnableLambda(execute_sql_query)
        )
    )

    # Step 3: Run the chain
    response = chain.invoke({"question": question})
    print("Chain Response:", response)

    return response

def get_schema_info():
    return db.get_table_info()

print(nl2sql_tool("I want to know the top ethnicities with the most part time employees"))