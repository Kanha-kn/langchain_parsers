from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# load the document
loader = TextLoader("docs.txt")
documents = loader.load()

# split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# convert text into embeddings and store in FASSI (vector DB)
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# create a retriver (this fetches relevent documents)
retriever = vectorstore.as_retriever()

# Initialize LLM
llm = OpenAI(model_name = "gpt-3.5-turbo",temperature=0.7)

# create chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)


# ask a question
query = "what are the key takeaways from the document?"
answer = qa_chain.run(query)

print("answer", answer)

