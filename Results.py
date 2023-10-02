# import argparse
# import joblib
# import numpy as np
# from keras.models import load_model
# from keras_contrib.layers import CRF
# from keras_contrib.losses import crf_loss
# from PyPDF2 import PdfFileReader  # Import PyPDF2 for PDF text extraction
# from keras_preprocessing.sequence import pad_sequences
# from keras_contrib.metrics import crf_viterbi_accuracy
# import os
#
#
# # Your other functions and imports here...
# def test_sentence_sample(test_sentence, word2idx, tags, max_len, model_predict):
#     results = []
#     x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]], padding="post",
#                                 value=0, maxlen=max_len)
#     p = model_predict.predict(np.array([x_test_sent[0]]))
#     p = np.argmax(p, axis=-1)
#     for w, pred in zip(tokenize(test_sentence), p[0]):
#         results.append([w, tags[pred]])
#     return results
#
#
# def tokenize(s):
#     if isinstance(s, (list, tuple)):
#         s = '\n'.join(map(str, s))
#     return s.split('\n')
#
#
# def post_processing(prediction_results):
#     b_descriptions = []
#     i_descriptions = []
#
#     for line, label in prediction_results:
#         if label == 'B-VARETEKST':
#             b_descriptions.append(line)
#         elif label == 'I-VARETEKST':
#             i_descriptions.append(line)
#
#     # Combine B and I descriptions and return the final list
#     descriptions = b_descriptions + i_descriptions
#     return descriptions
#
#     def main(pdf_folder_path):
#         # Your code that processes the PDF folder goes here
#         print(f"Processing PDFs in folder: {pdf_folder_path}")
#
#
# def main(pdf_file_path):
#     print(f"Processing PDF file: {pdf_file_path}")
#
#     model_predict = load_model(r'/home/sami/PycharmProject/Datascience/bi_lstm_weights_5/model.h5',
#                                custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
#                                                'crf_viterbi_accuracy': crf_viterbi_accuracy})
#
#     with open(r'/home/sami/PycharmProject/Datascience/bi_lstm_weights_5/words.pkl', 'rb') as f:
#         words = joblib.load(f)
#
#     with open(r'/home/sami/PycharmProject/Datascience/bi_lstm_weights_5/tags.pkl', 'rb') as f:
#         tags = joblib.load(f)
#
#     word2idx = {w: i + 1 for i, w in enumerate(words)}
#     tag2idx = {t: i for i, t in enumerate(tags)}
#     idx2tag = {i: w for w, i in tag2idx.items()}
#
#     max_len = 1000
#
#     # Extract text from the PDF using fitz
#     pdf_text = ""
#     with open(pdf_file_path, 'rb') as pdf_file:
#         pdf_reader = PdfFileReader(pdf_file)
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             pdf_text += page.extractText()
#
#     prediction_results = test_sentence_sample(pdf_text, word2idx, tags, max_len, model_predict)
#     print(prediction_results)
#     descriptions = post_processing(prediction_results)
#
#     # Print or return the results as needed
#     print(descriptions)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Predict text using a trained model')
#     parser.add_argument('pdf_file', type=str, help='Path to the PDF file')
#     args = parser.parse_args()
#     main(args.pdf_file)

import joblib
import numpy as np
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from PyPDF2 import PdfFileReader  # Import PyPDF2 for PDF text extraction
from keras_preprocessing.sequence import pad_sequences
from keras_contrib.metrics import crf_viterbi_accuracy
import os
import json


# Your other functions and imports here...

def test_sentence_sample(test_sentence, word2idx, tags, max_len, model_predict):
    results = []
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]], padding="post",
                                value=0, maxlen=max_len)
    p = model_predict.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    for w, pred in zip(tokenize(test_sentence), p[0]):
        results.append([w, tags[pred]])
    return results


def tokenize(s):
    if isinstance(s, (list, tuple)):
        s = '\n'.join(map(str, s))
    return s.split('\n')


def post_processing(prediction_results):
    b_descriptions = []
    i_descriptions = []

    for line, label in prediction_results:
        if label == 'B-VARETEKST':
            b_descriptions.append(line)
        elif label == 'I-VARETEKST':
            i_descriptions.append(line)

    # Combine B and I descriptions and return the final list
    descriptions = b_descriptions + i_descriptions
    return descriptions


def load_custom_model(model_path, custom_objects):
    model = load_model(model_path, custom_objects=custom_objects)
    return model


def main(pdf_file_path):
    print(f"Processing PDF file: {pdf_file_path}")

    model_dir = '/home/sami/PycharmProject/Datascience /Carlsberg'
    model_file = 'model.h5'
    words_file = 'words.pkl'
    tags_file = 'tags.pkl'

    model_path = os.path.join(model_dir, model_file)
    words_path = os.path.join(model_dir, words_file)
    tags_path = os.path.join(model_dir, tags_file)

    model_predict = load_model(model_path, custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                                           'crf_viterbi_accuracy': crf_viterbi_accuracy})

    with open(words_path, 'rb') as f:
        words = joblib.load(f)

    with open(tags_path, 'rb') as f:
        tags = joblib.load(f)

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}

    max_len = 1000

    # Extract text from the PDF using PyPDF2
    pdf_text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            pdf_text += page.extractText()

    prediction_results = test_sentence_sample(pdf_text, word2idx, tags, max_len, model_predict)
    print(prediction_results)
    descriptions = post_processing(prediction_results)

    # Print or return the results as needed
    print(descriptions)


if __name__ == "__main__":
    pdf_file_path = '/home/sami/PycharmProject/Datascience /BC_Catering/pdf/inv-0A6CQ-1648451449.pdf'  # Replace with the path to your PDF file
    main(pdf_file_path)
