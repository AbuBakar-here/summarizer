from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, string, openai
# import fitz, tiktoken, Keys
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from flask_cors import CORS, cross_origin


########### remove below text #######################

from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import textract


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

dbs = {}

####################################################


app = Flask(__name__, static_url_path='/static', static_folder='./static')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = 'pdfs-to-test'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

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

        # doc = fitz.open(complete_file_path)
        # content = ""
        # print(len(doc))
        # for page in doc:
        #     content += page.get_text()
        # content = remove_non_ascii(content)

        # converting pdf's text to tokens
        # enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # len_of_tokens = len(enc.encode(content))

        # if len_of_tokens > 4000:
        #     return {"success": False, "message": "Token limit exceed!"}


        ################### remove below code #################
        
        doc = textract.process(complete_file_path)
        with open('./pdfs-to-test/abc.txt', 'w') as f:
            f.write(doc.decode('utf-8'))

        with open('./pdfs-to-test/abc.txt', 'r') as f:
            content = f.read().replace("\n\n", "\n")

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 100,
            chunk_overlap  = 24,
            length_function = count_tokens
        )

        chunks = text_splitter.create_documents([content])
        # Get embedding model
        embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_KEY)

        # Create vector database
        db = FAISS.from_documents(chunks, embeddings)


        ########################################################

        # Save content to MongoDB
        doc = {"filename": filename, "type": filename.split('.')[-1], "content": content}
        res = collection.insert_one(doc)
        
        #### delete below line
        dbs[str(res.inserted_id)] = db

        return {"_id": str(res.inserted_id), "success": True, "error": False, "message": "contents of the file have been extracted."}
    
    return {"success": False, "error": True, "message": "either file is not pdf or file is not present"}


# # create a function to fetch pdf data from database and feed it to chatgpt
# @app.route('/ask-question', methods=['POST'])
# @cross_origin()
# def ask_question():
#     _id = request.form['_id']
#     question = request.form['question']

#     if not _id:
#         return {"success": False, "error": True, "message": "_id of the content is required"}
#     elif not question:
#         return {"success": False, "error": True, "message": "Question is required"}
    
#     object_id = ObjectId(_id)
#     context = collection.find_one(object_id)["content"]
#     response = chatGPT(context, question)

#     if response["success"] != True:
#         print(response)
#         return response

#     doc = {"user_message": question, "assistant_message": response.choices[0].message["content"], "docId": object_id}
#     res = collection2.insert_one(doc)
#     return response

################### remove below code ########################

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


# utility functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_non_ascii(a_str):
    a_str = a_str.replace("\n\n", "\n")
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def chatGPT(context, question, temperature=1):
  openai.api_key = OPENAI_KEY
  messages = [
    {"role": "system", "content": f"read below text, I will ask you questions from this:\n{context}"},
    {"role": "user", "content": question}
  ]
  try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
    )

    return { "success": True, "response": response }
  
  except Exception as e:
    print(f"error: {e}")
    return {"success": False, "message": str(e)}


########################### remove below code ##############################

def chatPDF(prompt, assistant, history=None):
  openai.api_key = OPENAI_KEY
  messages = [ 
      {"role": "system", "content": f"You need to answer my questions from below text. Try to answer me in a most descriptive way.\nText:\n{assistant}"}
      ]
  if len(history) > 0:
    messages.extend(history)
    print(messages[1:])

  messages.append({"role": "user", "content": f"{prompt}. Make this as descirptive as possible"})
  try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
    )
    return { "success": True, "response": response }
  except Exception as e:
    print(f"error: {e}")
    return {"success": False, "message": str(e)}

def docs_to_text(docs):
  text = ""
  for i in docs:
    text += i.page_content.replace("\n\n", "\n")
  
  return text

def get_chat_history(_id):
    chat_history = collection2.find_one( { "docId": ObjectId(_id) } )
    if chat_history:
        return chat_history["chatHistory"]
    return []

def update_chat_history(_id, updated_doc):
    response = collection2.update_one( { "docId": ObjectId(_id) }, {"$set": updated_doc}, upsert=True )

#########################################################################    


if __name__ == "__main__":
    app.run()



# add error handling for chatgpt