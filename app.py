from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def DocumentLoader():
    """
    This function loads a PDF document and splits it into smaller chunks.
    It uses the PyPDFLoader to load the document and RecursiveCharacterTextSplitter to split it.
    default chunk size and overlap is 1000 and 20 respectively
    you can change these values as per your requirement
    """

    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()

    #default values
    chunk_size = 1000
    chunk_overlap = 20


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    texts = text_splitter.split_documents(documents)

    print (texts)
    pass
