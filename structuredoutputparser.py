from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = "fact_1" , description="fact 1 about the topic"),
    ResponseSchema(name = "fact_2" , description="fact 2 about the topic"),
    ResponseSchema(name = "fact_3" , description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me  3 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({"topic":"black hole"})

print(result)
print(type(result))