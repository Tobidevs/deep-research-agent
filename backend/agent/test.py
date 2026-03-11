from langchain.chat_models import init_chat_model
from pydantic import BaseModel
import os

from agent.utils.state import ClarifyWithUser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="openai:gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
)


class Clarify(BaseModel):
    need_clarification: bool
    question: str


structured_output_model = model.with_structured_output(Clarify)

response = structured_output_model.invoke(
    [
        HumanMessage(
            content="""Answer the user question accurately and determine if you need to ask a follow up question to clarify the user's intent. 
                     If you need to ask a follow up question, return  with exact keys: "need_clarification": <boolean to determine if you need clarification>, "question": <the question you want to ask the user to clarify their intent>. 
                     User question: Who is in the world?"""
        )
    ]
)
print(response.need_clarification)
print(response.question)
