import os


def extract_doc_title(doc_url: str) -> str:
    filename = os.path.basename(doc_url)
    return os.path.splitext(filename)[0]
