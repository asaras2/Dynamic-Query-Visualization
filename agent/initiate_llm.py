from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_openai import ChatOpenAI
import os
from typing import Optional


def create_gpt_llm(api_key: Optional[str] = None) -> ChatOpenAI:
    """
    Create ChatOpenAI instance with provided API key or fallback to environment variable.
    
    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY from environment.
    
    Returns:
        ChatOpenAI instance configured with GPT-4o
    """
    if api_key:
        return ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    else:
        # Fallback to environment variable for development
        return ChatOpenAI(model="gpt-4o")