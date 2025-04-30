from dotenv import load_dotenv, find_dotenv
load_dotenv(override=True)
from langchain_community.utilities import SQLDatabase
import sqlalchemy
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from operator import itemgetter
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain.tools import tool
from pydantic import BaseModel
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from typing import TypedDict, Annotated, Sequence
import json, ast

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing_extensions import Literal

from agent.initiate_llm import llama_llm
import pandas as pd



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
db = setup_database_connection()

# 1a) tell LangChain “I want exactly one field called query”
response_schemas = [
    ResponseSchema(name="query", description="The final corrected SQL query in backticks.")
]

# 1b) build a parser that will enforce that shape
output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)


sql_chain_template = """
You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
{top_k}. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
For the given tables and their schemas, think about the question and what tables need to be used/joined in order to answer the question.
Do not limit yourself to only one table if the question requires columns from multiple tables, think about which tables should be joined.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".


Only use the following tables:
{table_info}
=====


Question: {input}

Use the following format for the output:
"""

format_inst = output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
sql_chain_template += "\n\n" + format_inst

# print(sql_chain_template)

# Create the PromptTemplate object
custom_sql_prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=sql_chain_template
)


sql_correction_template = """
You are a PostgreSQL expert.  
An earlier attempt at answering the user’s question failed.  
User Question:
{question}

Previous SQL:
{query}

Error message:
{error}

Table schema:
{table_info}

Based on the error, you may need to join other tables or fix column names.  
Please use the following output format:

"""

sql_correction_template += "\n\n" + format_inst

correction_prompt = PromptTemplate(
    input_variables=["question", "query", "error", "table_info"],
    template=sql_correction_template
)

correction_prompt = correction_prompt.partial(
    table_info=db.get_table_info(),
)

import re

def extract_sql_block(text):
    # print(text)
    match = re.search(r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.

    Args:
        text (str): Raw SQL query text that may contain code blocks, tags, and backticks

    Returns:
        str: Cleaned SQL query
    """
    # print("Raw SQL Query:", text)
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
    # print("Cleaned SQL Query:", text)

    return text

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

def nl2sql_chain():
    """Tool to Generate and Execute SQL Query to answer questions."""
    print("INSIDE NL2SQL TOOL")

    # Step 1: SQL generation chain
    write_query = create_sql_query_chain(llama_llm, db, prompt=custom_sql_prompt)

    # Step 2: Build the full runnable chain
    chain = (
        RunnablePassthrough.assign(
            query=write_query | RunnableLambda(lambda raw: output_parser.parse(raw)["query"])
        )
    )

    return chain

sql_chain = nl2sql_chain()
correction_chain = correction_prompt | llama_llm | output_parser


def gen_sql_node(state: AgentState):
    last_msg = state["messages"][-1]
    question = last_msg.content
    # print("\n\n=======IN SQL TOOL WITH ==>", last_msg.content)
    response_dict = sql_chain.invoke({"question": question})
    # print("\n>>>QUERY WAS ==> ", response_dict['query'])

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
    # print("\n\n========IN CHECK WITH ==>", data)
    result = data["result"]

    # if there's an error, ask the correction_chain to fix it
    if isinstance(result, str) and result.startswith("Error:"):
        correction = correction_chain.invoke({
            "question": data["question"],
            "query": data["query"],
            "error": result.split("'\n")[0]
        })

        updated_state = {
            "question": data["question"],
            "query": correction['query'],
        }
        
        # emit the corrected‐SQL message and go re‐run exec_sql
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

    # print("WE DONE")
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


#### Tool for making dataframe

def make_dataframe(query: str, result):
    """Convert SQL query result to a Pandas DataFrame."""
    
    df_schema = pd.read_sql(query, db._engine)
    # print(df_schema)
    return df_schema


initial_state = { "messages": [HumanMessage(content="Give me the names of 5 employees along with their full time or part time status")] }
final_state = graph.invoke(initial_state)
print(final_state)

SQL_SUBAGENT = graph

temp = [['Hispanic or Latino', 81], ['Native American', 57], ['Asian', 67], ['White', 74], [None, 144]]
df = make_dataframe('SELECT "ethnic_description", COUNT("employee_id") FROM analytical_schema.dim_ukg_employee_demographic_details GROUP BY "ethnic_description" LIMIT 5', temp)
print(df)
