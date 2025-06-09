import glob
import os

import ollama
from pypdf import PdfReader


def extract_text(pdf_path) -> tuple[str, str]:
    document_directory = os.path.dirname(pdf_path)
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text = text.strip()
    text_path = os.path.join(document_directory, 'text')
    if not os.path.exists(text_path):
        os.makedirs(os.path.join(document_directory, 'text'))
    file_name = os.path.basename(pdf_path).replace('.pdf', '.txt')
    with open(os.path.join(text_path, file_name), "w") as text_file:
        text_file.write(text)



if __name__ == '__main__':
    for file in glob.glob('/workspace/documents/**/**/*.pdf'):
        extract_text(file)