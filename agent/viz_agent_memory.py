from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(override=True)
from langgraph.types import Command
from typing_extensions import Literal
import pandas as pd

class code(BaseModel):
    """Schema for code solutions from the coding assistant"""
    imports: str = Field(description="Code Block import statements")
    code: str = Field(description="Code block not including import statements")

from agent.initiate_llm import gpt_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

code_system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a python coding assistant with expertise in exploratory data analysis and data visualization using Plotly.

You will be provided with:
- df_description: a textual description of the DataFrame’s columns and types.
- results: a list of tuples where each tuple is a row of data, matching df_description.
- question: a user query - could be the question asking about the data/updates on a previous result or just a query for a plot type.

Your task:
1. Inspect `question` for any chart-type preference (e.g. “line”, “bar”, “scatter”, “histogram”).
2. If no explicit preference:
   a. Examine column types:
      - One numeric + one categorical → bar chart or histogram.
      - Two numerics → scatter plot (or line if one is datetime).
      - Three columns (e.g., two numerics + one categorical) → grouped or stacked bar.
      - Multiple numeric columns + time series context → multi-line chart.
   b. Choose the best chart type based on data shape and semantics.
3. Define:
   - x-axis (typically the categorical or datetime column).
   - y-axis (numeric column(s); for multi-series, one trace per column).
   - A descriptive title that references the question.
4. Produce Python code that:
   - Imports all required modules (`plotly.graph_objects as go`, `pandas`, etc.).
   - Constructs a DataFrame from `results` and `df_description`.
   - Builds a Plotly figure object (e.g. `go.Figure(...)`).
   - Assigns `fig.update_layout(...)` for title and axes labels.
   - **Returns** the figure object (do **not** call `.show()`).

Ensure any code you provide is complete and executable as-is, with imports, data setup, and the final `fig` variable ready to display.

Then list:
1. **Imports**
2. **Code block** that returns the Plotly figure object.
""",
        ),
        (
            "placeholder", "{messages}"
        ),
        
    ]
)

code_agent_chain = code_system_prompt | gpt_llm.with_structured_output(code)

from agent.sql_react_agent_llama import df
import io
buf = io.StringIO()

# df.dtypes

question = "Give the number of employees for each ethnicity"
results = list(df.itertuples(index=False, name=None))
df.info(buf=buf, memory_usage=False, verbose=True)
df_desc = buf.getvalue()

# temp_messages = [
#             (
#                 "user",
#                 f"""
#                     The user has asked the following 'question': {question}.
#                     Here is the description of the dataframe - 'df_description': {df_desc}.
#                     Here is the data that is in the dataframe - 'results': {results}.
                    
#                     Invoke the code tool to structure the output with the prefix, imports and code block."""
#             )
# ]

# temp_messages

# res = code_agent_chain.invoke(
#         {"messages": temp_messages}
#     )

# print(res.imports)
# print(res.code)

from typing import List
from typing_extensions import TypedDict

class State(TypedDict):
    """
    Represents the state of the Graph.
    Attributes:
    error : Binary flag for control flow to check whether test error was tripped
    messages: chat history with user questions and AI responses
    generation: code solution

        
    """
    question: str
    df: pd.DataFrame
    results: List[tuple]
    error: str
    messages: List
    generation: str

# Define the nodes
def generate(state: State) -> Command[Literal["check_code"]]:

    """
    Node to generate code solution

    Arguments:
        state (dict): The current graph state

    Returns:
        state (dict): The updated graph state

    """

    print("----- Generating Code -----")

    # Store State variables
    messages = state["messages"]
    error = state["error"]

    question = state["question"]
    df = state["df"]
    results = state["results"]

    buf = io.StringIO()
    df.info(buf=buf, memory_usage=False, verbose=True)
    df_desc = buf.getvalue()    

    # if we have been routed back with error
    if error == "yes":
        # error fix prompt
        messages += [
            (
                "user",
                "Now, try again. Invoke the code tool to structure the output with the imports and code block."
            )
        ]
    
    else:
        messages+= [
            (
                "user",
                f"""
                    User Query - `question`: {question}.
                    Here is the description of the dataframe - 'df_description': {df_desc}.
                    Here is the data that is in the dataframe - 'results': {results}.
                    
                    Invoke the code tool to structure the output with the imports and code block."""
            )
        ]
    
    code_solution = code_agent_chain.invoke(
        {"messages": messages}
    )

    messages += [
            (
                "assistant",
                f"Imports: {code_solution.imports} \n Code Block: {code_solution.code}",
            )
        ]


    # if code_solution.prefix == "end":
    #     next_step = "end"
    # else:
    #     messages += [
    #         (
    #             "assistant",
    #             f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code Block: {code_solution.code}",
    #         )
    #     ]

    return Command(goto="check_code", 
    update =
        {
            "generation": code_solution,
            "messages": messages,
        }
    )
    
    # return {"generation": code_solution, "messages": messages, "iterations": iterations, "error_iterations": error_iterations, "next_step": next_step, "error": error}



from langgraph.graph import END, StateGraph, START

def execute_and_check_code(state: State) -> Command[Literal["generate", "__end__"]]:
    """
    Execute code and check for errors.

    Arguments:
        state (dict): The current Graph state.

    Returns:
        state (dict): The updated graph state.
    """

    print("----- Executing code -----")

    # storing state variables
    messages = state["messages"]
    code_solution = state["generation"]
    imports = code_solution.imports
    code_to_run = code_solution.code

    # check imports
    try:
        exec(imports)
    except Exception as e:
        print("---- Code Exceution Failed: Imports ----")
        error_message = [
            (
                "user",
                f"Your code solution failed the import test: {e}"
            )
        ]
        messages += error_message
        return Command(goto="generate",
        update = {
            "messages": messages,
            "error": "yes"
        }) 
    
    # check execution
    local_vars = {}
    try:
        exec(imports + "\n" + code_to_run, {}, local_vars)

    except Exception as e:
        print("---- Code Exceution Failed: Code Block ----")
        error_message = [
            (
                "user",
                f"Your code failed the code execution test: {e}"
            )
        ]
        messages += error_message
        return Command(goto="generate",
        update = {
            "messages": messages,
            "error": "yes"
        }) 
    
    # check if fig is in local vars
    if "fig" not in local_vars:
        print("---- Code Exceution Failed: No Figure Object ----")
        error_message = [
            (
                "user",
                f"Your code solution failed the test: no figure object was returned. Please make sure that you have a figure object in your code and that it is named 'fig'."
            )
        ]
        messages += error_message
        return Command(goto="generate",
        update = {
            "messages": messages,
            "error": "yes"
        })
    
    # No failures
    print("---- No Code Failures----")
    return Command(goto=END)

# def decide_to_finish(state: State):

#     """
#     Checks whether maximum allowed iterations are over
#     """

#     iterations = state["iterations"]
#     error_iterations = state["error_iterations"]
#     next_step = state["next_step"]

#     if iterations >= max_iters or error_iterations >= max_trys or next_step == "end":
#         print("---DECISION: FINISH---")
#         return "end"
#     else:
#         return "execute_code"

workflow_builder = StateGraph(State)

# define the nodes
workflow_builder.add_node("generate", generate) # generate solution
workflow_builder.add_node("check_code", execute_and_check_code) # execute and check code

# Build Graph
workflow_builder.add_edge(START, "generate")
# workflow.add_edge("generate", "check_code")
# workflow_builder.add_conditional_edges(
#     "generate",
#     decide_to_finish,
#     {
#         "end": END,
#         "execute_code": "check_code"
#     }
# )
# workflow_builder.add_edge("check_code", "generate")
workflow = workflow_builder.compile()

# from IPython.display import Image, display

# try:
#     display(Image(workflow.get_graph().draw_mermaid_png()))
# except Exception:
#     # You can put your exception handling code here
#     pass


initial_state = {"question": question, "df": df, "results": results, "messages":[], "error": ""}

solution = workflow.invoke(initial_state)

print(solution['generation'].imports)
print(solution['generation'].code)


VIZ_AGENT = workflow




