from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({"topic":"black hole"})

result= model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1= model.invoke(prompt2)

print(result1.content)

