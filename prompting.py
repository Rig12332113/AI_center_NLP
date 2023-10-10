from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = OpenAI(openai_api_key="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")
chat_model = ChatOpenAI(openai_api_key="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")
text = "早安"
Input = [HumanMessage(content=text)]
print(llm.predict_messages(Input).content)
print("----------------------------------------------")
print(chat_model.predict_messages(Input).content)
