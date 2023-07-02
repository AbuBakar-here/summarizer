from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import os#, Keys
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_cors import CORS, cross_origin
from utils.utils import *


########### remove below text #######################

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, HuggingFaceHubEmbeddings


# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v1")
# OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or Keys.OPENAI_API_KEY
# embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_KEY)
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")# or Keys.HUGGINGFACEHUB_API_TOKEN
embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L12-v1", huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN)
os.system("mkdir pdfs-to-test")
os.system("mkdir vector-store")

####################################################


app = Flask(__name__, static_url_path='/static', static_folder='./static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = 'pdfs-to-test'
VECTOR_STORE = 'vector-store'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# MongoDB configuration
db_user_name = "PythonPDF"
db_password = "PythonPDF"
client = MongoClient("mongodb+srv://{}:{}@cluster0.z2uh8u9.mongodb.net/?retryWrites=true&w=majority".format(db_user_name, db_password))
db = client['summarizer']
collection = db['pdf_data']
collection2 = db['messages']

@app.route('/', methods=['GET'])
def file_upload():
    return render_template('upload.html')


@app.route('/extract-content', methods=['POST'])
@cross_origin()
def extract_content():

    if 'file' not in request.files:
        return {"success": False, "error": True, "message": "No File Part"}

    file = request.files['file']

    if file.filename == '':
        return {"success": False, "error": True, "message": "No selected file"}

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        complete_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(complete_file_path)


        ################### remove below code #################
        
        # doc = textract.process(complete_file_path)
        # with open('./pdfs-to-test/abc.txt', 'w') as f:
        #     f.write(doc.decode('utf-8'))

        # with open('./pdfs-to-test/abc.txt', 'r') as f:
        #     content = f.read().replace("\n\n", "\n")

        # text_splitter = RecursiveCharacterTextSplitter(
        #     # Set a really small chunk size, just to show.
        #     chunk_size = 100,
        #     chunk_overlap  = 24,
        #     length_function = count_tokens
        # )

        # chunks = text_splitter.create_documents([content])
        
        chunks = process_documents(complete_file_path)
        
        if not chunks["success"]:
            return {"success": False, "error": True, "message": "either file is not pdf or file is not present"}
        chunks = chunks["chunks"]

        # Create vector database
        db = FAISS.from_documents(chunks, embeddings)


        ########################################################

        # Save content to MongoDB
        doc = {"filename": filename, "type": filename.split('.')[-1], "content": docs_to_text(chunks)}
        res = collection.insert_one(doc)
        
        # Save vector database
        db.save_local(os.path.join(VECTOR_STORE, str(res.inserted_id)))

        return {"_id": str(res.inserted_id), "success": True, "error": False, "message": "contents of the file have been extracted."}
    
    return {"success": False, "error": True, "message": "We do not support this format yet."}


@app.route('/generate-document-info', methods=['POST'])
@cross_origin()
def generate_document_info():
    _id = request.form['_id']
    if not _id:
        return {"success": False, "error": True, "message": "_id of the content is required"}
    
    question = "determine the topic from provided text then generate a 50-100 words summary of the text and then generate 3 questions from given text related to topic"
    docs = FAISS.load_local(os.path.join(VECTOR_STORE, _id), embeddings).similarity_search(question)
    text = docs_to_text(docs)
    ans = ""
    try:
        for line in chatPDF(question, text):
            if 'content' in line['choices'][0]['delta']:
                ans += line['choices'][0]['delta']['content']
        
        return {"success": True, "error": False, "content": ans}
    except Exception as e:
        print(f"error from generate_document_info: {e}")
        return {"success": False, "error": True, "message": str(e)}


@app.route('/ask-question', methods=['POST'])
@cross_origin()
def ask_question():
    _id = request.form['_id']
    question = request.form['question']

    if not _id:
        return {"success": False, "error": True, "message": "_id of the content is required"}
    elif not question:
        return {"success": False, "error": True, "message": "Question is required"}
    
    
    docs = FAISS.load_local(os.path.join(VECTOR_STORE, _id), embeddings).similarity_search(question)
    text = docs_to_text(docs)
    chat_history = get_chat_history(_id)
    # response = chatPDF(question, text, chat_history)
    
    def gen():
        ans = ""
        for line in chatPDF(question, text, chat_history):
            if 'content' in line['choices'][0]['delta']:
                ans += line['choices'][0]['delta']['content']
                yield line['choices'][0]['delta']['content']
        
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": ans})
        
        doc = {"chatHistory": chat_history, "docId": ObjectId(_id)}
        
        update_chat_history(_id, doc)
    
    return Response(gen(), mimetype="text/event-stream")

    # if response["success"] != True:
    #     print(response)
    #     return response

    # append new messages
    # chat_history.append({"role": "user", "content": question})
    # chat_history.append({"role": "assistant", "content": response["response"].choices[0].message["content"]})
    
    # doc = {"chatHistory": chat_history, "docId": ObjectId(_id)}
    
    # update_chat_history(_id, doc)
    
    # return response

##################################################################################

def get_chat_history(_id):
    chat_history = collection2.find_one( { "docId": ObjectId(_id) } )
    if chat_history:
        return chat_history["chatHistory"]
    return []

def update_chat_history(_id, updated_doc):
    response = collection2.update_one( { "docId": ObjectId(_id) }, {"$set": updated_doc}, upsert=True )

##################################################################################


if __name__ == "__main__":
    app.run(thread=True)