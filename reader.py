import pytesseract
import cv2
import numpy as np
import textstat
import spacy
from matplotlib import pyplot as plt
from pytesseract import Output
from PIL import Image
from flask import Flask, request, Response
from textstat.textstat import textstatistics, legacy_round
import json
import unidecode

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR-2\tesseract.exe'

def anti_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

##def returntext(image, boxes: bool):
def returntext(image):
    img = cv2.imread(image,0)
    nonoise = anti_noise(img)
    thresh = thresholding(nonoise)
    d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
##    if boxes:
##        n_boxes = len(d['text'])
##        for i in range(n_boxes):
##            if int(float(d['conf'][i])) > 60:
##                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
##                imgbox = cv2.rectangle(thresh, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('finished.jpg', thresh)
    fintext = pytesseract.image_to_string(Image.open('finished.jpg'))
##    escapes = ''.join([chr(char) for char in range(1,32)])
##    translator = str.maketrans('','',escapes)
##    fintext = fintext.translate(translator)
    fintext = unidecode.unidecode(fintext)
    fintext = fintext.replace("\n", " ")
    fkg = textstat.flesch_kincaid_grade(fintext)
    diff_words = difficult_words(fintext)
    dw = len(diff_words)
    out = { "text": fintext, "fkg": fkg, "dw": dw, "dw_set": diff_words}
    outj = json.dumps(out)
    return outj

def break_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)

def syllables_count(word):
    return textstatistics().syllable_count(word)
    
def difficult_words(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
    diff_words = []
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2 and len(word) >= 7:
            diff_words.append(word)
    return diff_words

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    data = request.files['image']
    ext = data.filename.split(".")[1]
    data.save("temp." + ext)
    print(data)
    response = Response(returntext("temp." + ext))
    response.headers["Access-Control-Allow-Origin"] = '*'
    return response
    
    
app.run('0.0.0.0', 5050)
