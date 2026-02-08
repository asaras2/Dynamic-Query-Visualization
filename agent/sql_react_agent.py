from dotenv import load_dotenv
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

from agent.initiate_llm import create_gpt_llm  # Changed to dynamic function
import pandas as pd


from IPython.display import Image, display



import os
from typing import Optional, Any, List, Union, cast

END_NODE: Literal["__end__"] = "__end__"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_counter: int

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

def create_sql_database(db_url: str, schema: Optional[str] = None) -> SQLDatabase:
    engine = sqlalchemy.create_engine(db_url)
    return SQLDatabase(engine, schema=schema) if schema else SQLDatabase(engine)

# Define a structured response schema as a Pydantic model
class SQLOutput(BaseModel):
    """The structured LLM output containing the SQL query."""
    sql_query: str = Field(description="The raw SQL query that answers the user's question")


def create_sql_llm_structured(api_key: Optional[str] = None):
    """Create structured LLM instance with user's API key."""
    gpt_llm = create_gpt_llm(api_key)
    return gpt_llm.with_structured_output(SQLOutput)



class SQLToolSchema(BaseModel):
    question: str

def make_dataframe(query: str, result: Any, *, db_url: str, schema: Optional[str] = None) -> pd.DataFrame:
    """Convert SQL query result to a Pandas DataFrame by re-running the query."""
    engine = sqlalchemy.create_engine(db_url)
    return pd.read_sql(query, engine)

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

def create_sql_agent_graph(api_key: Optional[str] = None, db_url: Optional[str] = None, schema: Optional[str] = None):
    """
    Create a SQL agent graph with user's API key.
    
    Args:
        api_key: User's OpenAI API key. If None, uses default/environment key.
    
    Returns:
        Compiled StateGraph for SQL generation and execution
    """
    if not db_url:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
        if not db_url:
            raise ValueError(
                "No Postgres connection URL provided. Pass db_url=... or set DATABASE_URL/POSTGRES_URL."
            )

    db = create_sql_database(db_url, schema=schema)
    table_info = db.get_table_info()

    def execute_sql_query(query: str) -> Union[str, Any]:
        execute_query = QuerySQLDatabaseTool(db=db)
        try:
            result = execute_query.run(query)
            if isinstance(result, str):
                try:
                    import ast
                    return ast.literal_eval(result)
                except Exception:
                    return result
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    # Create structured LLM with user's API key
    user_sql_llm_structured = create_sql_llm_structured(api_key)
    
    def gen_sql_node(state: AgentState):
        """Generate SQL query from the user's question."""
        last_msg = state["messages"][-1]
        question = last_msg.content

        system_msg = SystemMessage(content=build_system_sql_prompt(table_info))
        human_msg = HumanMessage(content=question)

        print("====SYSTEM PROMPT FOR SQL GENERATION====")
        print(system_msg.content)


        # Invoke structured LLM that returns SQLOutput directly
        response = cast(SQLOutput, user_sql_llm_structured.invoke([system_msg, human_msg]))
        
        response_dict = {"question": question, "query": response.sql_query}

        return {
            "messages": [
                ToolMessage(
                    content=json.dumps(response_dict),
                    name="nl2sql_tool",
                    tool_call_id="tool_123"
                )
            ],
            "retry_counter": 0,
        }

    def exec_sql_node(state: AgentState):
        last = state["messages"][-1]
        content = last.content if isinstance(last.content, str) else json.dumps(last.content)
        data = json.loads(content)
        # print("\n\n=======IN EXEC SQL NODE WITH ==>", data)
        sql_query = data['query']
        result = execute_sql_query(sql_query)
        print("\n\n=======SQL QUERY RESULT==>", result)

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
        content = last.content if isinstance(last.content, str) else json.dumps(last.content)
        data = json.loads(content)
        result = data["result"]
        print("\n\n=======IN CHECK NODE WITH SQL RESULT==>", result)

        retry_counter = int(state.get("retry_counter", 0)) + 1

        # if there's an error, call the structured LLM to produce a corrected SQL
        if isinstance(result, str) and result.startswith("Error:"):
            if retry_counter > 5:
                final_state = {
                    "question": data.get("question", ""),
                    "query": data.get("query", ""),
                    "result": (
                        "Error: SQL generation failed after 5 correction attempts. "
                        f"Last error was: {result}"
                    ),
                }
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=json.dumps(final_state),
                                name="exec_sql",
                                tool_call_id="tool_456",
                            )
                        ],
                        "retry_counter": retry_counter,
                    },
                    goto=END_NODE,
                )

            system_msg = SystemMessage(content=build_system_correction_prompt(table_info))
            human_content = (
                f"User Question:\n{data['question']}\n\nPrevious SQL:\n{data['query']}\n\n"
                f"Error message:\n{result}"
            )
            human_msg = HumanMessage(content=human_content)

            # Use structured LLM for correction
            corrected_response = cast(SQLOutput, user_sql_llm_structured.invoke([system_msg, human_msg]))
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
                ], "retry_counter": retry_counter},
                goto="exec_sql"
            )

        # otherwise we're done
        return Command(update={"retry_counter": retry_counter}, goto=END_NODE)

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("gen_sql", gen_sql_node)
    builder.add_edge("gen_sql","exec_sql")
    builder.add_node("exec_sql", exec_sql_node)
    builder.add_edge("exec_sql","check")
    builder.add_node("check", check_node)
    builder.set_entry_point("gen_sql")
    
    return builder.compile(name="sql_agent")

# visulize the graph with mermaid
# display(Image(graph.get_graph().draw_mermaid_png()))

# # save the image to a file
# with open("sql_react_agent_llama_graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())


#### (legacy make_dataframe removed; use the keyword-only version above)
