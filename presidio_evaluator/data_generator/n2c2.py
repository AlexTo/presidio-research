from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import spacy

import sys

if __name__ == '__main__':
    folder = sys.argv[1]
    nlp = spacy.load("en_core_web_trf")
    xmls = []

    for file_name in tqdm(os.listdir(folder)):
        if not file_name.endswith(".xml") or "_" in file_name:
            continue
        with open(os.path.join(folder, file_name), encoding='utf-8') as f:
            xmls.append({"file_name": file_name,
                         "content": BeautifulSoup(f.read(), features="xml")})

    docs = list(nlp.pipe([xml["content"].deIdi2b2.TEXT.text for xml in xmls]))
