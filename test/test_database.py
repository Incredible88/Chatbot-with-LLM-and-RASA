from langchain.document_loaders.mongodb import MongodbLoader
import os,json
from pymongo import MongoClient
import asyncio
import aiomysql
import numpy as np
from bson import json_util
import re
import sqlalchemy
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain.docstore.document import Document
from langchain.document_loaders.mongodb import MongodbLoader

# Set API Key
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-4',temperature = 0.1)

connection_string = "mongodb://localhost:27017/"
db_name = "HSG"
collection_name = "bank_info"
def create_documents_from_json(data, parent_key=''):
    documents = []

    if isinstance(data, dict):
        for key, value in data.items():
            # Skip the "_id" key at the top level
            if key == '_id':
                continue

            nested_key = f'{parent_key}.{key}' if parent_key else key
            documents.extend(create_documents_from_json(value, nested_key))
    elif isinstance(data, list):
        for item in data:
            documents.extend(create_documents_from_json(item, parent_key))
    else:
        content = f"{parent_key}: {json.dumps(data)}" if parent_key else json.dumps(data)
        document = Document(page_content=content, metadata={"key": parent_key})
        documents.append(document)

    return documents

def extract_text_from_json_recursive(data, indent=""):
    text = ""

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                text += f"{indent}{key}:\n"
                text += extract_text_from_json_recursive(value, indent + "  ")
            else:
                text += f"{indent}{key}: {value}\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                text += extract_text_from_json_recursive(item, indent + "  ")
            else:
                text += f"{indent}- {item}\n"
    else:
        text += f"{indent}{data}\n"

    return text

def extract_keywords_tfidf(docs, max_features=50):

    if len(docs) == 1:
        docs = [sentence.strip() for sentence in docs[0].split('.') if sentence]

    # Create and configure the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english', max_features=None)

    # Train the model
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)

    # Retrieve the vocabulary
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Aggregate the scores for each feature
    aggregated_scores = np.sum(tfidf_matrix.toarray(), axis=0)

    # Sort indices in descending order of aggregated_scores
    sorted_indices = np.argsort(aggregated_scores)[::-1]

    # Extract top features based on sorted_indices, ensuring we don't exceed the number of available features
    top_n = min(max_features, len(feature_names))
    top_features = feature_names[sorted_indices[:top_n]]

    refined_keywords = [word for word in top_features if re.search('[a-zA-Z]', word)]

    return set(refined_keywords)

def create_documents_from_dataframe(df, source_path):
    documents = []
    for index, row in df.iterrows():
        # Generate page content with column names as prefix
        page_content = '\n'.join([f'{col}: {row[col]}' for col in df.columns])
        # Create a Document with this content and metadata
        document = Document(page_content=page_content,
                            metadata={'source': source_path, 'row': index})
        documents.append(document)
    return documents

async def load_mongodb_data_async(connection_string, db_name, collection_name):
    # 创建一个 MongodbLoader 实例
    loader = MongodbLoader(connection_string, db_name, collection_name)

    # 在不同的线程中运行同步的 load 方法
    loop = asyncio.get_event_loop()
    bank_info = await loop.run_in_executor(None, loader.load)

    return bank_info

async def load_from_mysql():
    connection_params = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'Czw200513',
        'db': 'banking_chatbot',
    }

    async with aiomysql.create_pool(**connection_params) as pool:
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("SELECT * FROM transactions")
                records = await cur.fetchall()
                return pd.DataFrame(records)

# loader = MongodbLoader(connection_string, db_name, collection_name)

# docs = loader.load()
# documents = []
#
# documents.extend(docs)
script_dir = os.path.dirname(__file__)
file_path_j = os.path.join(script_dir, '..', 'data', 'client_data.json')

# # Load JSON documents
with open(file_path_j, 'r') as json_file:
    json_data1 = json.load(json_file)
json_document1 = create_documents_from_json(json_data1)
# documents.extend(json_document)

#
# client = MongoClient('localhost', 27017)
# db = client['HSG']
# collection = db['bank_info']
#
# documents = list(collection.find({}))
#
# documents_dict = {str(doc['_id']): doc for doc in documents}
# json_text = extract_text_from_json_recursive(documents_dict)
#
# a = 0
#
# json_text1 = extract_text_from_json_recursive(json_data)
#
# b =1
# docs_json = [json_text]
# docs_json1 = [json_text1]
# keywords_json = extract_keywords_tfidf(docs_json)
#
# keywords_json1 = extract_keywords_tfidf(docs_json1)

# bank_info = loader.load()
# documents.extend(bank_info)
#
# engine = sqlalchemy.create_engine("mysql://root:Czw200513@localhost:3306/banking_chatbot")
# sql_query = "SELECT * FROM transactions"
# df_sql = pd.read_sql_query(sql_query, engine)
# sql_documents = create_documents_from_dataframe(df_sql,"transactions")
# documents.extend(sql_documents)


documents = []

# load mongoDB
connection_string = "mongodb://localhost:27017/"
db_name = "HSG"
collection_name = "bank_info"
client = MongoClient(connection_string)
db = client[db_name]
collection = db[collection_name]
documentdb = list(collection.find())
json_data = json.dumps(documentdb, default=json_util.default)
dict_data = json.loads(json_data, object_hook=json_util.object_hook)[0]
json_document = create_documents_from_json(dict_data)
documents.extend(json_document)
client.close()
# loader = MongodbLoader(connection_string, db_name, collection_name)
# bank_info = loader.load()
# documents.extend(bank_info)

# load MySQL
engine = sqlalchemy.create_engine("mysql://root:Czw200513@localhost:3306/banking_chatbot")
sql_query = "SELECT * FROM transactions"
df_sql = pd.read_sql_query(sql_query, engine)
sql_documents = create_documents_from_dataframe(df_sql,"transactions")
documents.extend(sql_documents)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5)
docs = text_splitter.split_documents(documents)

# define embedding
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)

# define retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
