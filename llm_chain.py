from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# load the LLM (GPT-3.5)
llm = OpenAI(model_name = "gpt-3.5-turbo",temperature=0.7)

# create a prompt template

prompt = PromptTemplate(
    input_variables=["topic"],
    template="suggest a catchy blog type about {topic}"
)

chain = LLMChain(llm=llm,prompt=prompt)

topic = input("enter a topic")

output = chain.run(topic)

print("Generated Blog Title", output)