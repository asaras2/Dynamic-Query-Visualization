from dotenv import load_dotenv, find_dotenv
load_dotenv(override=True)
from langchain_community.utilities import SQLDatabase
import sqlalchemy
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_community.tools import QuerySQLDatabaseTool
from operator import itemgetter
import re

from langchain.tools import tool
from pydantic import BaseModel, Field

from typing import TypedDict, Annotated, Sequence
import json, ast

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing_extensions import Literal

from agent.initiate_llm import gpt_llm
import pandas as pd


from IPython.display import Image, display



import os
import urllib

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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

# Initialize database connection
db = setup_database_connection()

# Pre-fill values (no prompt templates or partials)
table_info = db.get_table_info()
top_k = "Return the most informative results."

# Define a structured response schema as a Pydantic model
class SQLOutput(BaseModel):
    """The structured LLM output containing the SQL query."""
    sql_query: str = Field(description="The raw SQL query that answers the user's question")


# Create structured LLM that returns SQLOutput
sql_llm_structured = gpt_llm.with_structured_output(SQLOutput)



class SQLToolSchema(BaseModel):
    question: str

# Helper function for executing cleaned SQL queries
def execute_sql_query(query: str) -> str:
    """Execute SQL query against the database."""
    execute_query = QuerySQLDatabaseTool(db=db)
    try:
        result = execute_query.run(query)
        # print("Query Execution Result:", result)

        if isinstance(result, str):
            try:
                import ast
                return ast.literal_eval(result)
            except Exception:
                return result
        return result

    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        # print(error_msg)
        return error_msg

# Note: we no longer build a separate SQL chain with prompt templates.
# Instead, nodes call the raw `sql_llm` and parse outputs with `PydanticOutputParser`.
def build_system_sql_prompt(table_info: str) -> str:
    """Return the system instruction text used to prompt the LLM for SQL generation."""
    return f"""
You are a PostgreSQL expert. Generate a syntactically correct PostgreSQL SQL query to answer the user's question.
You can order the results to return the most informative data in the database.
IMPORTANT:
- Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
- Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
- For the given tables and their schemas, think about the question and what tables need to be used/joined in order to answer the question.
- Do not limit yourself to only one table if the question requires columns from multiple tables, think about which tables should be joined.
- Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".
- Look at all tables and all columns in the database schema to identify the relevant tables and columns needed to answer the question.


### PostgreSQL Database Schema
{table_info}
""".strip()


def build_system_correction_prompt(table_info: str) -> str:
    """Prompt for correcting SQL when an error occurs."""
    return f"""
You are a PostgreSQL expert. The previous SQL query failed with an error. Correct the query based on the error message and the database schema.
Based on the error, you may need to join other tables or fix column names, or correct a misformed SQL query.

### PostgreSQL Database Schema
{table_info}
""".strip()


def gen_sql_node(state: AgentState):
    """Generate SQL query from the user's question."""
    last_msg = state["messages"][-1]
    question = last_msg.content

    system_msg = SystemMessage(content=build_system_sql_prompt(table_info))
    human_msg = HumanMessage(content=question)

    # Invoke structured LLM that returns SQLOutput directly
    response = sql_llm_structured.invoke([system_msg, human_msg])
    
    response_dict = {"question": question, "query": response.sql_query}

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(response_dict),
                name="nl2sql_tool",
                tool_call_id="tool_123"
            )
        ]
    }

def exec_sql_node(state: AgentState):
    last = state["messages"][-1]
    data = json.loads(last.content)
    # print("\n\n=======IN EXEC SQL NODE WITH ==>", data)
    sql_query = data['query']
    result = execute_sql_query(sql_query)

    updated_state = {
        "question": data["question"],
        "query": sql_query,
        "result": result
    }


    return {
        "messages": [
            ToolMessage(
                content=json.dumps(updated_state),
                name="exec_sql",
                tool_call_id="tool_456"
            )
        ]
    }

def check_node(state: AgentState) -> Command[Literal["exec_sql","__end__"]]:
    last = state["messages"][-1]
    data = json.loads(last.content)
    result = data["result"]

    # if there's an error, call the structured LLM to produce a corrected SQL
    if isinstance(result, str) and result.startswith("Error:"):
        system_msg = SystemMessage(content=build_system_correction_prompt(table_info))
        human_content = (
            f"User Question:\n{data['question']}\n\nPrevious SQL:\n{data['query']}\n\n"
            f"Error message:\n{result}"
        )
        human_msg = HumanMessage(content=human_content)

        # Use structured LLM for correction
        corrected_response = sql_llm_structured.invoke([system_msg, human_msg])
        corrected_query = corrected_response.sql_query

        updated_state = {
            "question": data["question"],
            "query": corrected_query,
        }
        
        # emit the corrected SQL message and go re-run exec_sql
        return Command(
            update={"messages":[
                ToolMessage(
                    content=json.dumps(updated_state),
                    name="correct_sql",
                    tool_call_id="tool_789"
                )
            ]},
            goto="exec_sql"
        )

    # otherwise we're done
    return Command(goto=END)

builder = StateGraph(AgentState)
builder.add_node("gen_sql", gen_sql_node)
builder.add_edge("gen_sql","exec_sql")
builder.add_node("exec_sql", exec_sql_node)
builder.add_edge("exec_sql","check")
builder.add_node("check", check_node)
builder.set_entry_point("gen_sql")
graph = builder.compile(name="sql_agent")

# visulize the graph with mermaid
# display(Image(graph.get_graph().draw_mermaid_png()))

# # save the image to a file
# with open("sql_react_agent_llama_graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())


#### Tool for making dataframe

def make_dataframe(query: str, result):
    """Convert SQL query result to a Pandas DataFrame."""
    
    df_schema = pd.read_sql(query, db._engine)
    # print(df_schema)
    return df_schema

# initial_state = { "messages": [HumanMessage(content="give the average salary of employees each year for the past 5 years")] }
# final_state = graph.invoke(initial_state)
# for m in final_state["messages"]:
#     m.pretty_print()

SQL_SUBAGENT = graph

temp = [['Hispanic or Latino', 81], ['Native American', 57], ['Asian', 67], ['White', 74], [None, 144]]
df = make_dataframe('SELECT "ethnic_description", COUNT("employee_id") FROM analytical_schema.dim_ukg_employee_demographic_details GROUP BY "ethnic_description" LIMIT 5', temp)
# print(df)
