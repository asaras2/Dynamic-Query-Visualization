import os
import urllib.parse
import re
import mysql.connector
from openai import OpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool  # Updated import
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Set NVIDIA API Key
os.environ["OPENAI_API_KEY"] = "api_key"

# Database credentials
db_user = "root"
db_password = "password"
db_host = "localhost"
db_name = "chinook"

# URL-encode the password
encoded_password = urllib.parse.quote(db_password, safe="")

# Connection URI
connection_uri = f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}/{db_name}"
print("Connecting to:", connection_uri)

# Create SQLDatabase instance
db = SQLDatabase.from_uri(connection_uri)

# Print database details
print("SQL Dialect:", db.dialect)

# Initialize NVIDIA NIM Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Generate SQL Query Prompt
user_question = "How many genres are there?"
prompt = f"Generate a SQL query to answer: {user_question}. The database has the following tables:\n\n{db.get_table_info()}"

# Call NVIDIA API
response = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024
)

# Extract SQL Query from API response
sql_query = response.choices[0].message.content.strip()
match = re.search(r"sql\n([\s\S]*?);", sql_query)
sql_query = match.group(1).strip()
# if match:
#     return match.group(1).strip()
# raise ValueError("Failed to generate SQL query.")

print("\n\nGenerated SQL Query:", sql_query)

# cursor = conn.cursor()



# sql_query = """SELECT COUNT(GenreId) AS Number_of_Genres
# FROM chinook.genre;"""
# Execute the generated SQL query
execute_query = QuerySQLDatabaseTool(db=db)
query_result = execute_query.invoke(sql_query)

print("Query Result:", query_result)

# Define the Answer Rephrasing Prompt
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in natural language.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Answer: """
)

# Call NVIDIA API to rephrase the answer
rephrased_response = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[
        {
            "role": "user",
            "content": answer_prompt.format(
                question=user_question,
                query=sql_query,
                result=query_result,
            ),
        }
    ],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

# Extract the rephrased answer from the API response
rephrased_answer = rephrased_response.choices[0].message.content.strip()
print("\nFinal Answer:", rephrased_answer)
