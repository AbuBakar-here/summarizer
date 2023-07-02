from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os, openai#, Keys
import pandas as pd

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")# or Keys.OPENAI_API_KEY
ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'enex', 'epub', 'html', 'md', 'pdf', 'ppt', 'pptx', 'csv', "xls", "xlsx"}


def ExcelLoader(file_path: str, **kwargs):
  df = pd.read_excel(file_path, engine="openpyxl")
  new_file_path = file_path.rsplit(".", 1)[0] + ".csv"
  df.to_csv(new_file_path)
  return CSVLoader(new_file_path)


LOADER_MAPPING = {
    ".xls": (ExcelLoader, {}),
    ".xlsx": (ExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def chatPDF(prompt, assistant, history=[]):
  openai.api_key = OPENAI_KEY
  messages = [ 
      {"role": "system", "content": f"Search the answer of my questions from the 'Text' I am going to provide and answer me. Try to answer me in a most descriptive way.\nText:\n{assistant}"}
      ]
  if len(history) > 0:
    messages.extend(history[-15:])

  messages.append({"role": "user", "content": f"{prompt}. Make this as descirptive as possible"})
  try:
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=1,
        stream=True
    )
    
    # for line in response:
    #   if 'content' in line['choices'][0]['delta']:
    #     yield line['choices'][0]['delta']['content']
        
    # print(response)
    # return { "success": True, "response": response }

  except Exception as e:
    print(f"error: {e}")
    return {"success": False, "message": str(e)}

def docs_to_text(docs):
  text = ""
  for i in docs:
    text += i.page_content.replace("\n\n", "\n")
  
  return text
    
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))    

def load_single_document(file_path: str) -> list:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    return []

def process_documents(file_path: str) -> dict:
    """
    Load documents and split in chunks
    """
    documents = load_single_document(file_path)
    if not len(documents) < 1:
        {"success": False, "error": "Document is empty"}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=24, length_function = count_tokens)
    chunks = text_splitter.split_documents(documents)
    return {"success": True, "chunks": chunks}