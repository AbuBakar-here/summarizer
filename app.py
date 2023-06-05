from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, string, fitz, openai, Keys
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime


app = Flask(__name__, static_url_path='/static', static_folder='./static')
UPLOAD_FOLDER = 'pdfs-to-test'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
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

        doc = fitz.open(complete_file_path)
        content = ""
        for page in doc:
            content += page.get_text()
        content = remove_non_ascii(content)
        # open('pdf.txt', "w").write(content)


        # Save content to MongoDB
        doc = {"filename": filename, "type": filename.split('.')[-1], "content": content}
        res = collection.insert_one(doc)

        return {"_id": str(res.inserted_id), "success": True, "error": False, "message": "contents of the file have been extracted."}
    
    return {"success": False, "error": True, "message": "redirected"}


# create a function to fetch pdf data from database and feed it to chatgpt
@app.route('/ask-question', methods=['POST'])
def ask_question():
    _id = request.form['_id']
    print(_id)
    question = request.form['question']

    if not _id:
        return {"success": False, "error": True, "message": "_id of the content is required"}
    elif not question:
        return {"success": False, "error": True, "message": "Question is required"}
    
    object_id = ObjectId(_id)
    context = collection.find_one(object_id)["content"]
    response = chatGPT(context, question)
    doc = {"user_message": question, "assistant_message": response.choices[0].message["content"], "docId": object_id}
    res = collection2.insert_one(doc)
    return response

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
  openai.api_key = os['OPENAI_API_KEY']
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
    return response
  
  except Exception as e:
    return f"error: {e}"

# def extract_text_from_pdf(pdf_path):
#     pdf_reader = PyPDF2.PdfReader(open(pdf_path, "rb")).pages
#     len_of_pdf = len(pdf_reader)

#     # if pdf file have more than 200 pages use PyPDF2 else use pdfminer
#     if len_of_pdf > 200:
#         text = ""
#         for page_num in range(len_of_pdf):
#             page = pdf_reader[page_num]
#             text += page.extract_text()
#         return text

#     return pdf2txt.main(pdf_path).data

    

if __name__ == "__main__":
    app.run()



# add error handling for chatgpt