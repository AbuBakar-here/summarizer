from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
# import fitz, tiktoken, Keys
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_cors import CORS, cross_origin
from utils.utils import *#allowed_file, chatPDF, docs_to_text, get_chat_history, update_chat_history, count_tokens, load_single_document, process_documents, ExcelLoader


########### remove below text #######################

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v1")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or Keys.OPENAI_API_KEY
# embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_KEY)

dbs = {}

####################################################


app = Flask(__name__, static_url_path='/static', static_folder='./static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = 'pdfs-to-test'
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
        
        #### delete below line
        dbs[str(res.inserted_id)] = db

        return {"_id": str(res.inserted_id), "success": True, "error": False, "message": "contents of the file have been extracted."}
    
    return {"success": False, "error": True, "message": "We do not support this format yet."}


@app.route('/ask-question', methods=['POST'])
@cross_origin()
def ask_question():
    _id = request.form['_id']
    question = request.form['question']

    if not _id:
        return {"success": False, "error": True, "message": "_id of the content is required"}
    elif not question:
        return {"success": False, "error": True, "message": "Question is required"}
    
    
    docs = dbs[_id].similarity_search(question)
    text = docs_to_text(docs)
    chat_history = get_chat_history(_id)
    response = chatPDF(question, text, chat_history)

    if response["success"] != True:
        print(response)
        return response

    # append new messages
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response["response"].choices[0].message["content"]})
    
    doc = {"chatHistory": chat_history, "docId": ObjectId(_id)}
    
    update_chat_history(_id, doc)
    
    return response

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
    app.run()



# add error handling for chatgpt