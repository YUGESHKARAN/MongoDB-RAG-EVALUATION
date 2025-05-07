import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from mongodb_database import MongoDBDatabase
from langchain_groq import ChatGroq
import os
from langchain_anthropic import ChatAnthropic

import joblib
from sentence_transformers import  util

from langsmith import evaluate, Client
load_dotenv()

# Set API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['MONGODB_URI'] = os.getenv('MONGODB_URI')
claude_api = os.getenv('CLAUDE-API-KEY')

app = Flask(__name__)

CORS(app)

db = MongoDBDatabase("mongodb+srv://yugeshkaran01:GEMBkFW5Ny5wi4ox@blog.adtwl.mongodb.net/Blog-Data?retryWrites=true&w=majority&appName=blog", "Blog-Data")

chat_history = []

def mogodb_query_generator(db):
    template = """
You are a data analyst of a blog website. You are interacting with a user who is asking you questions about the blog app's database.
Based on the collection schema below, write a MongoDB query that would answer the user's question. Take the Conversation History into account.

<SCHEMA>{schema}</SCHEMA>

Conversation History: {chat_history}

Write only the MongoDB query and nothing else. Do not wrap the query in any other text, not even backticks.

For example:
Question: Show all the data?
MongoDB Query: collection.find{{}}

Question: total numeber of authors?
MongoDB Query:collection.distinct("authorname")

Question: total number of categories?
MongoDB Query: collection.distinct("posts.category")

Question: about the post Dimensionality Reduction?
MongoDB Query: collection.find({{'posts.title': 'Dimensionality Reduction'}})

Question: List of posts posted by author Yugesh Karan?
MongoDB Query:collection.find({{'authorname': "Yugesh Karan" }})

Question: display all the categories
MongoDB Query:collection.distinct("posts.category")

Question: how many followers author ajayvarsanr have?
MongoDB Query:collection.distinct("followers", {{"authorname": "ajayvarsanr"}})

Question: how many followers author ajayvarsan2020@gmail.com have?
MongoDB Query:collection.distinct("followers", {{"email": "ajayvarsan2020@gmail.com"}})

Question: how many post author ajayvarsanr posted?
MongoDB Query:collection.find({{'authorname': "ajayvarsanr"}}, {{'posts': 1, '_id': 0}})

Question: When last post was posted and name the author?
MongoDB Query: collection.find({{}}, {{'posts.timestamp': 1, 'authorname': 1, '_id': 0}}).sort({{'posts.timestamp': -1}}).limit(1)

Question: Posts on GenAI category?
MongoDB Query: collection.find({{ 'posts.category': 'GenAI' }})

Question: name the author who published post in both Data Science and GenAI categories?
MongoDB Query: collection.find({{"posts.category": {{"$in": ["Data Science", "GenAI"]}}}})



Note:
- With the help of schema generate a executable MongoDB Query without any error and mustt not add length() or count() method anywhere in the query, i.e `collection.find({{'authorname': "Pradeep"}}, {{'posts': 1, '_id': 0}}).count()` or `collection.find({{'authorname': "Pradeep"}}, {{'posts': 1, '_id': 0}}).length()` and must not warp query with string and should not use findOne() method insted use, collection.find({{}}, {{'authorname': 1, 'password': 0, 'email': 0, 'profile': 0, 'otp': 0, 'otpExpiresAt': 0, 'followers': 0, '__v': 0}}).
- Do not escape characters like underscores (`_`) or slashes (`/`) in names, emails, description, category or any other data. eg:'posts.description': '/Unsupervised learning/i' or  'posts.description': /.*Supervised Learning.*/i .
- collection name is collection  not db.collection.
- Do not add length() and count() attribute to the query, i.e. no `collection.distinct("authorname").length()` or `collection.distinct("followers", {{"authorname": "ajayvarsanr"}}).length()` or `collection.count({{'authorname': 'ajayvarsanr', 'posts': {{'$ne': []}}}})` or `collection.distinct("followers", {{"authorname": "ajayvarsanr"}}).length()` or `collection.distinct("posts", {{"authorname": "ajayvarsanr"}}).length()`.
- Provide the data exactly as given without adding unnecessary characters.
- correct format for generating projection is {{'authorname': 'haricharan_1133', 'posts.title': 'Computer Vision'}}, {{'posts.$': 1, '_id': 0}} so you must take this fromat as your primary reference.
- should not generate the query in the format of {{'authorname': 'haricharan_1133' }}, {{'posts.title': 'Computer Vision', 'posts.$': 1, '_id': 0 }} or collection.find({{'authorname': "ajayvarsanr"}}, {{'posts': 1, '_id': 0}}).count().
- should not generate the query in the format of collection.distinct("authorname").length() or collection.distinct("followers", {{"authorname": "ajayvarsanr"}}).length().
- correct format for generating the total  is collection.distinct("authorname") or collection.distinct("posts.category") or collection.distinct("posts") or collection.distinct("followers", {{"authorname": "ajayvarsanr"}}).
- Do not use `.count()`, i.e. no collection.find({{'authorname': "ajayvarsanr"}}, {{'posts': 1, '_id': 0}}).count() or any other query with count().
- Follow all the above instruction and look the example Question and MongoDB Query before generating the query.

Your turn:
Question:{question}
MongoDB Query:
"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", anthropic_api_key=claude_api)


    return (
        RunnablePassthrough.assign(
            schema=lambda _: db.get_collection_schema('authors')
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def response_generator(user_query: str, db: MongoDBDatabase, chat_history: list):
    # Generate the MongoDB query using the query generator
    mongo_chain = mogodb_query_generator(db)
    template = """
    You are a copilot for a blog website. You are interacting with a chief who is asking you questions about the blog's database to generate content, statistics measure or required infromation from the database based on user query.
    Based on the collection schema below, cheif question, MongoDB query, and MongoDB response, write a natural language response with pre-size. 
    note:
    1. Generate the content as per the conversation history and MongoDB response.
    2. Make sure to format the response as paragraph.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    MongoDB Query: <QUERY>{query}</QUERY>
    User Question: {question}
    MongoDB Response: {response}

    If the MongoDB response is not empty, confirm the existence of the post and author and show the post content.
    If the MongoDB response is empty, inform the user that the post wasn't found or suggest alternative searches.
    """


    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", anthropic_api_key=claude_api)

    # Create the prompt with the template
    prompt = ChatPromptTemplate.from_template(template)

    # Define the chain
    chain = (
        RunnablePassthrough.assign(query=mongo_chain)  # Ensure query is passed correctly
        .assign(
            schema=lambda _: db.get_collection_schema('authors'),
            response=lambda var: db.run('authors',var['query'])  # Ensure query is passed as a dictionary
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Execute the chain and return the result
    return chain.invoke({"question": user_query, "chat_history": chat_history})


@app.route("/")
def welcome_blog_backend():
    return "Welcome to Blog Browser"

@app.route("/query-rag",methods=['POST'])
def query_MongoDB_RAG():
    
    data = request.json
    user_query = data.get("query","")
    if user_query:
        chat_history.append(HumanMessage(content=user_query))

        response = response_generator(user_query, db, chat_history)

        if response:
            chat_history.append(AIMessage(content=response))

        return jsonify({"response":response})

if __name__ == "__main__":
    app.run(port=4001, host="0.0.0.0", debug=False)




# 1. Create and/or select your dataset
client = Client()
dataset_name = "MongoDB Database"
model = joblib.load("sbert_model.pkl")


def get_flask_response(x):
    print(f"Evaluating input: {x}")  # Debugging line

    # Check for 'question' instead of 'query'
    query =x.get("question")  # Support both keys
    if not query:
        print("Error: 'query' or 'question' key is missing in input data!")
        return {"output": ""}  # Prevent crash

    # Send request to Flask API
    response = requests.post("http://127.0.0.1:4001/query-rag", json={"query": query})
    
    try:
        response_json = response.json()
        print(f"Received response: {response_json}")  # Debugging line
        return response_json
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"output": ""}


# 2. Define an evaluator
def exact_match(outputs: dict, reference_outputs: dict) -> bool:
    return outputs == reference_outputs

def accuracy(outputs: dict, reference_outputs: dict) -> float:
    """Computes similarity between obtained and reference outputs."""
    output_text = outputs.get("response", "")
    reference_text = reference_outputs.get("output", "")

    if not output_text or not reference_text:
        print("Warning: Missing output text, returning 0 similarity.")
        return 0.0  # Avoid crashing if output is missing

    # Compute similarity
    model = joblib.load("sbert_model.pkl")
    output_embedding = model.encode(output_text, convert_to_tensor=True)
    reference_embedding = model.encode(reference_text, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(output_embedding, reference_embedding).item()
    return similarity

# Compute error rate
def error_rate(outputs: dict, reference_outputs: dict) -> float:
    """Calculates error rate based on empty or missing responses."""
    output_text = outputs.get("response", "")

    if not output_text:  # If no response or empty output
        return 1.0  # 100% error for this instance
    return 0.0  # No error if a valid response is present

# 4️⃣ Correctness Evaluator (Binary Threshold)
def Correctness(outputs: dict, reference_outputs: dict) -> int:
    """Checks if similarity is above 0.69 threshold."""
    similarity_score = accuracy(outputs, reference_outputs)
    return 1 if similarity_score > 0.69 else 0

# To evaluate a LangGraph graph, replace lambda with graph.invoke
evaluate(
    get_flask_response,  # Fetch correct Flask output
    data=dataset_name,
    evaluators=[exact_match, accuracy, error_rate, Correctness],
    experiment_prefix="claude-3-7-sonnet-20250219"
)

