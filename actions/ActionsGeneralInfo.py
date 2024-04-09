import json
import os
import re
import PyPDF2
import numpy as np
import pandas as pd
import sqlalchemy
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from langchain.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import  ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, JSONLoader, CSVLoader
from langchain.docstore.document import Document

# Set API Key
os.environ['OPENAI_API_KEY'] = 'YOUR-KEY'
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-4',temperature = 0.1)

dialogues = [
    {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "I'm your personal banking assistant. How can I assist you today?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "I'm your personal banking assistant. How can I assist you today?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Can you help me with Math or Physics?"},
        {"role": "assistant", "content": "Sorry, I am a banking assistant chatbot, can only help with banking staffs."}
    ]},
    {"messages": [
        {"role": "user", "content": "Can you tell me my account balance?"},
        {"role": "assistant", "content": "Sure, could you please specify which account you’d like to check? For example, save or pay?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Save account, please."},
        {"role": "assistant", "content": "Your save account balance is $5,000. Is there anything else I can assist you with?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Show me my last transactions."},
        {"role": "assistant", "content": "Your most recent transaction was Transaction No. 5000010 on 2024/10/14, lt was a debit transaction for -200 CHF at BCC restaurant. After this transaction, your balance in the Pay account was 2.000　CHF．　Would you like to see more transactions or need help with something else?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Show me my last 2　transactions."},
        {"role": "assistant", "content": """Your most recent　2 transaction as follow:
            1. Transaction Number: No. 5000010
              Date: 2024/10/14
              Change: -200 CHF
              Balance after: 2,000 CHF
              Information: BCC restaurant bills

            2. Transaction Number: No. 5000009
              Date: 2024/10/13
              Change: -60 CHF
              Balance after: 2,200 CHF
              Information: COOP City cost
        Would you like to see more transactions or need help with something else?"""}
    ]},
    {"messages": [
        {"role": "user", "content": "No, thank you."},
        {"role": "assistant", "content": "You're welcome! If you have more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is my account number?"},
        {"role": "assistant", "content": "Sure, Your account number is [Account Number]. Would you like to see more transactions or need help with something else?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thank you."},
        {"role": "assistant", "content": "You're welcome! If you have more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is the interest rate on a savings account?"},
        {"role": "assistant", "content": "Our current interest rate for a savings account is 2% APY. Do you have other questions?"}
    ]},
    {"messages": [
        {"role": "user", "content": "What is the interest rate on 1 year deposit?"},
        {"role": "assistant", "content": "Our current interest rate for 1-year time deposit is 3 per cent. Do you have other questions?"}
    ]},
    {"messages": [
        {"role": "user", "content": "Thanks! Bye!"},
        {"role": "assistant", "content": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!"}
    ]}
]

def extract_text_from_pdf(pdf_path):
    # Initialize a variable for text
    text = ""

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            # Get a page
            page = pdf_reader.pages[page_num]

            # Extract text
            text += page.extract_text()

    return text
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

def create_documents_from_json(data, parent_key=''):
    documents = []

    if isinstance(data, dict):
        for key, value in data.items():
            nested_key = f'{parent_key}.{key}' if parent_key else key
            documents.extend(create_documents_from_json(value, nested_key))
    elif isinstance(data, list):
        for item in data:
            documents.extend(create_documents_from_json(item, parent_key))
    else:
        # Combine key and value into page_content
        content = f"{parent_key}: {json.dumps(data)}" if parent_key else json.dumps(data)
        document = Document(page_content=content, metadata={"key": parent_key})
        documents.append(document)

    return documents

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


script_dir = os.path.dirname(__file__)
# file_path_d = os.path.join(script_dir, '..', 'data', 'data.pdf')
# file_path_t = os.path.join(script_dir, '..', 'data', 'Transactions.csv')
file_path_j = os.path.join(script_dir, '..', 'data', 'client_data.json')


# load pdf file
# loader = PyPDFLoader(file_path_d)
# document = loader.load()
# documents.extend(document)
def qaRetrival():
    documents = []
    # Load JSON documents
    with open(file_path_j, 'r') as json_file:
        json_data = json.load(json_file)
    json_document = create_documents_from_json(json_data)
    documents.extend(json_document)

    # load MySQL
    engine = sqlalchemy.create_engine("YOUR-MYSQL")
    sql_query = "SELECT * FROM transactions"
    df_sql = pd.read_sql_query(sql_query, engine)

    sql_documents = create_documents_from_dataframe(df_sql,"transactions")
    documents.extend(sql_documents)

    # csv_loader = CSVLoader(file_path_t)
    # csv_document = csv_loader.load()
    # documents.extend(csv_document)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5)
    docs = text_splitter.split_documents(documents)

    # define embedding
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

    return retriever, memory
    # Define QA LLM model


# Keywords classification

# Extract keywords From database
engine = sqlalchemy.create_engine("YOUR-MYSQL")
sql_query = "SELECT * FROM transactions"
df_sql = pd.read_sql_query(sql_query, engine)
text_column = df_sql.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Create and configure the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english', max_features=None)

# Train the model
tfidf_matrix = tfidf_vectorizer.fit_transform(text_column)

# Retrieve the vocabulary
feature_names = list(tfidf_vectorizer.vocabulary_.keys())

# Filter out keywords with digits
refined_keywords_db = [word for word in feature_names if not re.search(r'\d', word)]

# Extract keywords From json
with open(file_path_j, 'r') as json_file:
    json_data = json.load(json_file)
json_document = create_documents_from_json(json_data)
json_text = extract_text_from_json_recursive(json_data)
docs_json = [json_text]
keywords_json = extract_keywords_tfidf(docs_json)

# Merge the keywords from the database and json
keywords = set(refined_keywords_db).union(keywords_json)

common_greetings = set(["hi", "hello", "thank you", "thanks", "goodbye", "bye", "Yes", "No", "OK", "Sure","Transaction","Transactions"])

# Define classification function
def is_question_relevant(question: str, keywords: set, common_phrases: set) -> bool:
    # Combine the keywords and common phrases
    all_relevant_words = keywords.union(common_phrases)

    # Check if the question contains any of the relevant words
    return any(word.lower() in question.lower() for word in all_relevant_words)

