# import PyPDF2
# import pickle
# import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
#
# # Load the pre-trained model and tokenizers
# model = tf.keras.models.load_model('model.h5')  # Load your model here
# with open('words.pkl', 'rb') as words_file:
#     words = pickle.load(words_file)
# with open('tags.pkl', 'rb') as tags_file:
#     tags = pickle.load(tags_file)
#
#
# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_file_path):
#     pdf_text = ""
#     with open(pdf_file_path, 'rb') as pdf_file:
#         pdf_reader = PyPDF2.PdfFileReader(pdf_file)
#         for page_num in range(pdf_reader.numPages):
#             page = pdf_reader.getPage(page_num)
#             pdf_text += page.extract_text()
#     return pdf_text
#
#
# # Function to make predictions on the extracted text
# def make_predictions(text, MAX_SEQUENCE_LENGTH=1000):
#     # Tokenize and preprocess the text (replace this with your own tokenization and preprocessing)
#     # Example tokenization and preprocessing:
#     text = text.lower()
#     # ... perform tokenization and preprocessing steps specific to your model ...
#
#     # Convert text to numerical sequences
#     word_ids = [words.get(word, 0) for word in text.split()]
#
#     # Pad the sequence to match model input shape
#     padded_word_ids = pad_sequences([word_ids], padding='post', maxlen=MAX_SEQUENCE_LENGTH)
#
#     # Make predictions
#     predictions = model.predict(padded_word_ids)
#     # Post-process predictions (e.g., convert to tags)
#     # ...
#
#     return predictions
#
#
# # Example usage
# pdf_file_path = '/home/sami/PycharmProject/Datascience /BC_Catering/pdf/inv-0A6CQ-1648451449.pdf'  # Replace with the path to your PDF file
# pdf_text = extract_text_from_pdf(pdf_file_path)
# predictions = make_predictions(pdf_text)
#
# # Print or process the predictions as needed
# print(predictions)

import PyPDF2
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model and tokenizers
model = tf.keras.models.load_model('model.h5')  # Load your model here
with open('words.pkl', 'rb') as words_file:
    words = pickle.load(words_file)
with open('tags.pkl', 'rb') as tags_file:
    tags = pickle.load(tags_file)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    pdf_text = ""
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            pdf_text += page.extract_text()
    return pdf_text

# Function to make predictions on the extracted text
def make_predictions(text):
    # Tokenize and preprocess the text (replace this with your own tokenization and preprocessing)
    # Example tokenization and preprocessing:
    text = text.lower()
    # ... perform tokenization and preprocessing steps specific to your model ...

    # Convert text to numerical sequences
    word_ids = [words.get(word, 0) for word in text.split()]

    # Pad the sequence to match model input shape
    padded_word_ids = pad_sequences([word_ids], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    # Make predictions
    predictions = model.predict(padded_word_ids)
    # Post-process predictions (e.g., convert to tags)
    # ...

    return predictions

# Example usage
pdf_file_path = '/home/sami/PycharmProject/Datascience /BC_Catering/pdf/inv-0A6CQ-1648451449.pdf'  # Replace with the path to your PDF file
pdf_text = extract_text_from_pdf(pdf_file_path)
predictions = make_predictions(pdf_text)

# Print or process the predictions as needed
print(predictions)
