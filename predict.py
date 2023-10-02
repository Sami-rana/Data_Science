import argparse
import joblib
import numpy as np
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from pdfminer.high_level import extract_text
from keras_preprocessing.sequence import pad_sequences
from keras_contrib.metrics import crf_viterbi_accuracy


# def test_sentence_sample(test_sentence, word2idx, tags, max_len, model_predict):
#     results = []
#     x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]], padding="post",
#                                 value=0, maxlen=max_len)
#     p = model_predict.predict(np.array([x_test_sent[0]]))
#     p = np.argmax(p, axis=-1)
#     for w, pred in zip(tokenize(test_sentence), p[0]):
#         results.append([w, tags[pred]])
#     return results

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

    def main(pdf_folder_path):
        # Your code that processes the PDF folder goes here
        print(f"Processing PDFs in folder: {pdf_folder_path}")
# def main():
#     parser = argparse.ArgumentParser(description='Predict text using a trained model')
#     parser.add_argument('pdf_file', type=str, help='Path to the PDF file')
#     args = parser.parse_args()

    model_predict = load_model(r'/home/sami/PycharmProject/Datascience /BC_Catering/bi_lstm_weights_5/model.h5',
                               custom_objects={"CRF": CRF, 'crf_loss': crf_loss,
                                               'crf_viterbi_accuracy': crf_viterbi_accuracy})

    # Load the word2idx and tag2idx dictionaries
    with open(r'/home/sami/PycharmProject/Datascience /BC_Catering/bi_lstm_weights_5/words.pkl', 'rb') as f:
        words = joblib.load(f)

    with open(r'/home/sami/PycharmProject/Datascience /BC_Catering/bi_lstm_weights_5/tags.pkl', 'rb') as f:
        tags = joblib.load(f)

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}

    max_len = 1000

    pdf_text = extract_text(args.pdf_file)
    # print(pdf_text)
    prediction_results = test_sentence_sample(pdf_text, word2idx, tags, max_len, model_predict)
    print(prediction_results)
    descriptions = post_processing(prediction_results)

    # Print or return the results as needed
    print(descriptions)


# if __name__ == "__main__":
#     main()
