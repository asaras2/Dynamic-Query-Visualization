from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_openai import ChatOpenAI


gpt_llm = ChatOpenAI(model="gpt-4o")