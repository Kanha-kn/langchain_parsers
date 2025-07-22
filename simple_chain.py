from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt = PromptTemplate(
    template = "generate 5 interesting facts abot {topic}",
    input_variables=["topic"]
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

chain = prompt | model | parser

result = chain.invoke({"topic":"cricket"})

print(result)

chain.get_graph().print_ascii()