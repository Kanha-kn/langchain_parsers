from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#1st prompt
template1 = PromptTemplate(
    template='write a dtailed report on {topic}',
    input_variables=["topic"]
)

# 2nd prompt

template2 = PromptTemplate(
    template='write a 5 line summary on the following text: \n {text}',
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"black hole"})

print(result)