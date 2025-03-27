import os
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Database connection parameters
SERVER_NAME = ""
DATABASE_NAME = ""
USERNAME = ""
PASSWORD = ""

# Llama Model Connection Parameters
LLAMA_API_URL = ""
LLAMA_API_KEY = ""

# Database connection URI
db_uri = f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER_NAME}/{DATABASE_NAME}?driver=ODBC+Driver+17+for+SQL+Server"
db = SQLDatabase.from_uri(db_uri)

# Custom Llama API Wrapper
class LlamaAPIWrapper:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        
    def __call__(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama",
            "messages": [
                {"role": "system", "content": "You are a helpful SQL query generator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Llama API: {e}")
            return ""

# Initialize Llama API Wrapper
llm = LlamaAPIWrapper(LLAMA_API_URL, LLAMA_API_KEY)

# Template for SQL query generation
template = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question : {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)

def get_Schema(_):
    return db.get_table_info()

# SQL query generation chain
sql_chain = (
    RunnablePassthrough.assign(schema=get_Schema)
    | prompt
    | llm
    | StrOutputParser()
)

# Natural language response template
response_template = """
Based on the table schema below, question, sql query, and sql response, write a natural language response: 
{schema}

Question : {question}
SQL Query : {query}
SQL Response : {response}
"""

response_prompt = ChatPromptTemplate.from_template(response_template)

def run_query(query):
    return db.run(query)

# Full chain for generating SQL query and natural language response
full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_Schema,
        response=lambda variable: run_query(variable["query"])
    )
)

# Example usage
def main():
    try:
        # Example query
        result = full_chain.invoke({"question": "Give me overview of the database tables."})
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()