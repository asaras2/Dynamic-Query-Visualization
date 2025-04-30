from dotenv import load_dotenv
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, AzureMLEndpointApiType, CustomOpenAIChatContentFormatter
load_dotenv(override=True)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

llama_llm =  AzureMLChatOnlineEndpoint(
    endpoint_url="https://Meta-Llama-3-1-70B-Instruct-zhlk.westus.models.ai.azure.com/chat/completions",
    endpoint_api_type=AzureMLEndpointApiType.serverless,
    endpoint_api_key="", # Replace with your key
    content_formatter=CustomOpenAIChatContentFormatter(),
    model_kwargs={"temperature": 0.3, "max_tokens": 2000}
)

gpt_llm = ChatOpenAI(model="gpt-4o")
