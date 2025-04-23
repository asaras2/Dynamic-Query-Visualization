# Langgraph-Supervisor-SQL-Agent: Clean Code Folder

This folder contains the core components of the Langgraph-Supervisor-SQL-Agent project. Below is a description of each file and its purpose:

## Files

### 1. `initiate_llm.py`
This file initializes the language models (LLMs) used in the project, such as GPT and LLaMA. It provides the necessary configurations and setups for the LLMs to be used across the application.

### 2. `sql_react_agent_llama.py`
This file implements the SQL React Agent, which is responsible for:
- Generating SQL queries based on user questions.
- Executing the generated SQL queries.
- Handling errors in SQL execution and providing corrections.
- Utilizing LangChain and LangGraph libraries to create a structured workflow for SQL query generation and execution.

### 3. `supervisor_dynamic_viz.ipynb`
This Jupyter Notebook serves as the supervisor node for orchestrating the workflow. It:
- Routes tasks between different agents (`sql_agent`, `make_table_node`, `viz_agent`).
- Manages the overall state of the system using the `AgentState` class.
- Generates visualizations based on the results of SQL queries.
- Provides a dynamic and interactive environment for testing and debugging the system.

### 4. `viz_agent.py`
This file defines the visualization agent, which:
- Generates Python code for visualizing data using libraries like Plotly.
- Processes the DataFrame and user queries to create meaningful visualizations.
- Ensures the generated code adheres to best practices for data visualization.

