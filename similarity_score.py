from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
docs=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]
query="who is virat kohli"
embeded_docs=embedding.embed_documents(docs)
embedded_query=embedding.embed_query(query)
scores=cosine_similarity([embedded_query],embeded_docs)[0]
index,score=sorted(enumerate(scores),key=lambda x:x[1])[-1]
print(docs[index])

