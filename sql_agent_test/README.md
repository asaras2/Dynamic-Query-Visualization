# SQL Agent with Langchain and Langraph

This repository contains two implementations of an SQL agent using OpenAI's models:

1. **`make_sql_agent_executor.ipynb`** – Uses Langchain's SQL agent executor to process and execute SQL queries.
2. **`make_sql_agent_langraph.ipynb`** – Uses Langraph to define an AI workflow for handling SQL queries.

## Why Use Langraph?
The Langraph-based implementation is **preferred** because it allows for connecting multiple agents in a structured workflow, making it more scalable and adaptable.

## Setup Instructions

### 1. Create a `.env` File
Before running the notebooks, create a `.env` file in the project directory and add the following:

```ini
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
SERVER="your-database-server"
DATABASE="your-database-name"
USERNAME="your-username"
PASSWORD="your-password"
```
Replace the placeholders with your actual credentials.

### 2. Install Dependencies
Run the following command to install all required Python libraries:

```
pip install -r requirements.txt
```