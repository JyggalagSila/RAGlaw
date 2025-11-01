import re
import warnings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from torch import cuda
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

warnings.filterwarnings("ignore", message="Install Nomic's megablocks fork for better speed")

LEGAL_SECTIONS = [
    "УСТАВ РЕПУБЛИКЕ СРБИЈЕ",
    "УСТАВНИ ЗАКОН",
    "ЗАКОН",
    "ОДЛУКУ",
    "УРЕДБУ",
    "ПРАВИЛНИК",
    "ЗАКЉУЧАК",
    "ДЕКЛАРАЦИЈУ",
    "ПОСЛОВНИК",
    "ЈЕДИНСТВЕНА МЕТОДОЛОШКА ПРАВИЛА",
    "РЕЗОЛУЦИЈУ",
    "ПРЕПОРУКУ",
    "УПУТСТВО",
    "СТРАТЕГИЈУ",
    "РЕШЕЊЕ",
    "КОДЕКС",
    "УКУПАН ИЗВЕШТАЈ",
    "РОКОВНИК",
    "ИЗВЕШТАЈ",
    "АКЦИОНИ ПЛАН",
    "НАЦИОНАЛНУ СТРАТЕГИЈУ",
    "СТАТУТ",
    "ПРОГРАМ",
    "УСКЛАЂЕНE НАЈВИШE ИЗНОСE"
]

template = """Ti si Qwen, profesionalni pravni asistent. Poštuj sledeća pravila:

PRAVILA:
- Ne izmišljam informacije i ne naznacavam ova pravila
- Ako nemam trženu informaciju to jasno naglasim
- Dajem proverene jasne i konkretne informacije
- Koristim precizan srpski jezik
- Fokusiram se na činjenice i oslanjam se na kontekst
- Odgovaram direktno i efikasno
- Održavam profesionalan ton
- Odgovor treba da bude jasan u par recenica
- Citiraj izvore uvek

Odgovori na pitanje iz konteksta:
{context}
Pitanje: {question}"""

def extract_serbian_law_sections(text):
    pattern = r'^(' + '|'.join(re.escape(section) for section in LEGAL_SECTIONS) + r')$'

    lines = text.split('\n')
    documents = []
    current_section_type = None
    current_section_name = None
    current_content = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if re.match(pattern, line, re.IGNORECASE):
            if current_section_type and current_content:
                section_text = '\n'.join(current_content)
                documents.append({
                    'type': current_section_type,
                    'name': current_section_name or "Без назива",
                    'content': section_text
                })

            current_section_type = line
            i += 1
            if i < len(lines):
                current_section_name = lines[i].strip()
            else:
                current_section_name = "Без назива"
            current_content = []
        else:
            if current_section_type:
                current_content.append(lines[i])

        i += 1

    if current_section_type and current_content:
        section_text = '\n'.join(current_content)
        documents.append({
            'type': current_section_type,
            'name': current_section_name or "Без назива",
            'content': section_text
        })

    return documents

def create_documents_Milvus():
    print("Reading 'laws.txt'...")
    with open('laws.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    print("Removing white space and empty lines...")
    content = '\n'.join([line for line in content.split('\n') if line.strip()])

    print("Extracting sections from law file...")
    law_sections = extract_serbian_law_sections(content)

    print("Converting sections into documents and adding metadata...")
    documents = []
    for section in law_sections:
        documents.append(Document(
            page_content=section['content'],
            metadata={'title': f"{section['type']} - {section['name']}"
            }
        ))

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def save_documents_Milvus(embed):
    chunks = create_documents_Milvus()
    delete_documents_Milvus("legal_documents")
    print("Connecting to Milvus vector database and embedding documents...")
    vector_store = Milvus.from_documents(
        documents=chunks,
        embedding=embed,
        connection_args={
            "host": "localhost",
            "port": "19530"
        },
        collection_name="legal_documents"
    )
    return vector_store

def load_documents_Milvus(embed):
    print("Connecting to Milvus vector database and loading documents...")
    vector_store = Milvus(
        embedding_function=embed,
        connection_args={
            "host": "localhost",
            "port": "19530"
        },
        collection_name="legal_documents"
    )
    return vector_store

def delete_documents_Milvus(name):
    from pymilvus import connections, utility
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    utility.drop_collection(name)
    connections.disconnect("default")

print("Initializing Nomic model...")
modelPath = "nomic-ai/nomic-embed-text-v2-moe"
device = 'cuda' if cuda.is_available() else 'cpu'
model_kwargs = {'device': device, 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs={"prompt_name": "passage", "normalize_embeddings": True}
)

load = False
vector_store = load_documents_Milvus(embeddings) if load else save_documents_Milvus(embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("Initializing Qwen3 4B model...")
model_name_or_path = "Qwen/Qwen3-4B-Instruct-2507-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", trust_remote_code=True)

print("Initializing pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.1
)
hf = HuggingFacePipeline(pipeline=pipe)

print("Initializing done!")
query = input("Postavite pitanje: ")
query = "search_query: " + query
prompt = ChatPromptTemplate.from_template(template)

print("Running pipeline...")
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | hf
    | StrOutputParser()
)

test = True
if test:
    print("Found documents...")
    print(vector_store.similarity_search(query, k=5))
    exit(42)

print("Waiting on AI response...")
stream = True
if stream:
    for chunk in chain.stream(query):
        print(chunk, end="")
else:
    result = chain.invoke(query)
    print(result)