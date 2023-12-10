import datetime
import os
import re
import json

import mysql
import openai
import pandas as pd
from langchain.document_loaders import JSONLoader
from rasa.shared.nlu.training_data.message import Message
import numpy as np
from langchain.utilities import SQLDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.docstore.document import Document
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

import sqlalchemy

engine = sqlalchemy.create_engine("mysql://root:Czw200513@localhost:3306/banking_chatbot")

sql_query = "SELECT * FROM transactions"

df = pd.read_sql_query(sql_query, engine)

# mysql = SQLDatabase.from_uri("mysql://root:Czw200513@localhost:3306/banking_chatbot")
# sql_query = "SELECT * FROM transactions"
# sql_documents = mysql.run(sql_query)
# # script_dir = os.path.dirname(__file__)
# # file_path_t = os.path.join(script_dir, '..', 'data', 'Transactions.csv')
# # df = pd.read_csv(file_path_t, header=None)
#
# # 创建DataFrame对象（不需要列名）
# df = pd.DataFrame.from_records(sql_documents, columns=mysql.get_table_info().split("\n")[0].split(","))

text_column = df.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
# Create and configure the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english', max_features=None)

# Train the model
tfidf_matrix = tfidf_vectorizer.fit_transform(text_column)

# Retrieve the vocabulary
feature_names = list(tfidf_vectorizer.vocabulary_.keys())

# Filter out keywords with digits
refined_keywords= [word for word in feature_names if not re.search(r'\d', word)]


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


documents = create_documents_from_dataframe(df,"mysql")


print(documents)

amount = float(100)
recipient = "Alice"

config = {
    'user': 'root',
    'password': 'Czw200513',
    'host': 'localhost',
    'database': 'banking_chatbot',
}

# Connect to the database
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

try:
    # Update sender's account balance (Bob)
    cursor.execute(
        "UPDATE Accounts SET AccountBalance = AccountBalance - %s WHERE AccountName = 'Bob'",
        (amount,)
    )

    # Update recipient's account balance (Alice)
    cursor.execute(
        "UPDATE Accounts SET AccountBalance = AccountBalance + %s WHERE AccountName = %s",
        (amount, recipient)
    )

    # Get the new balance of Bob's account for the transaction record
    cursor.execute("SELECT AccountBalance FROM Accounts WHERE AccountName = 'Bob'")
    new_balance = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    number_of_transactions = cursor.fetchone()[0]

    # Insert a new transaction
    transaction_number = number_of_transactions + 1
    cursor.execute(
        "INSERT INTO transactions (transaction_number, date, amount_change, balance_after, receiver, information) VALUES (%s, %s, %s, %s, %s,%s)",
        (transaction_number, datetime.today().strftime('%Y-%m-%d'), -amount, new_balance, recipient, f"Bank transfer to {recipient}")
    )
    connection.commit()

except mysql.connector.Error as err:
    # Handle errors and rollback changes
    connection.rollback()


finally:
    # Close the connection
    cursor.close()
    connection.close()