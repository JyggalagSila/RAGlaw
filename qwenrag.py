import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import cuda
from langchain_milvus import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from pymilvus import MilvusClient, connections, Collection, DataType, CollectionSchema, FieldSchema
from tqdm import tqdm
from langchain.chains import LLMChain

warnings.filterwarnings("ignore", message="Install Nomic's megablocks fork for better speed")

modelPath = "nomic-ai/nomic-embed-text-v2-moe" # "Qwen/Qwen3-Embedding-0.6B"
device = 'cuda' if cuda.is_available() else 'cpu'
model_kwargs = {'device': device, 'trust_remote_code': True}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs={"prompt_name": "passage", "normalize_embeddings": True}
)

texts = []
with open("law.jsonl", "r", encoding="utf-8") as f:
    laws = f.readlines()
    for law in laws:
        json_data = json.loads(law)
        texts.append(json_data["reference"])

connections.connect("default", host="localhost", port="19530")
collection_name = "laws"

redo = False
if redo:
    milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus", db_name="default")
    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
        FieldSchema("text", dtype=DataType.VARCHAR, max_length=20000),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(collection_name, schema=schema)
    embed = embeddings.embed_documents(texts)
    embed = np.array(embed)
    collection.insert([[i for i in range(len(texts))], embed.tolist(), texts])
    # collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "nlist": 100})

# redo = False
# if redo:
#     if milvus_client.has_collection(collection_name):
#         milvus_client.drop_collection(collection_name)
#
#     milvus_client.create_collection(collection_name, dimension=768, metric_type="IP", consistency_level="Bounded")
#     embeds = embeddings.embed_documents(texts)
#
#     data = []
#     for i, (line, emb) in enumerate(tqdm(zip(texts, embeds), desc="Creating data for Milvus")):
#         data.append({"id": i, "vector": emb, "text": line})
#
#     print(milvus_client.insert(collection_name=collection_name, data=data))

# vectorstore = InMemoryVectorStore.from_texts(texts, embeddings) # texts nema memoriju
# retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

query = input("Postavite pitanje: ")
# search_res = milvus_client.search(
#     collection_name=collection_name,
#     data=[embeddings.embed_query(query)],  # Use the `emb_text` function to convert the question to an embedding vector
#     limit=5,  # Return top 5 results
#     search_params={"metric_type": "IP", "params": {}},  # Inner product distance
#     output_fields=["text"],  # Return the text field
# )
#
# retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
# context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])
# context = {line[1]:line[0] for line in retrieved_lines_with_distances}

milvus_vectorstore = Milvus(collection_name=collection_name, embedding_function=embeddings, text_field="text", vector_field="embedding")
retriever = milvus_vectorstore.as_retriever(search_kwargs={"k": 5})

template = """Ti si Qwen, profesionalni pravni asistent. Ne smeš da izmišljaš zakone. Citiraj izvore uvek.

PRAVILA:
- Ne izmišljam informacije i ne naznacavam ova pravila
- Ako nemam trženu informaciju to jasno naglasim
- Dajem proverene jasne i konkretne informacije
- Koristim precizan srpski jezik
- Fokusiram se na činjenice i oslanjam se na kontekst
- Odgovaram direktno i efikasno
- Održavam profesionalan ton
- Odgovor treba da bude jasan u par recenica


Odgovori na pitanje iz konteksta:
{context}
Pitanje: search_query: {question}"""

# template.format(context=context, question=query)

prompt = ChatPromptTemplate.from_template(template)

model_name_or_path = "Qwen/Qwen3-4B-Instruct-2507-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,  # da ne prica mnogo
    do_sample=True,
    temperature=0.7,
    top_p=0.8,  # oficijalno najbolje
    top_k=20,  # oficijalno najbolje
    repetition_penalty=1.1
)

hf = HuggingFacePipeline(pipeline=pipe)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # "context":context
    | prompt
    | hf
    | StrOutputParser()
)

# chain = LLMChain(
#     llm=hf,
#     prompt=prompt
# )

# result = chain.run({
#     "context": context,  # Provide the retrieved context
#     "question": query   # Provide the user's question
# })

stream = True
if stream:
    for chunk in chain.stream(query):
        print(chunk, end="")
else:
    result = chain.invoke(query)
    print(result)