import os
import io
import re
import fitz
import time
import sys
import json
import cv2
import glob
import joblib
import torch
import pandas as pd
import uvicorn
import tempfile
import string
import uvicorn
import tempfile
import pytesseract
import websockets
import subprocess
import pytesseract
import numpy as np
from typing import List, Union
from collections import OrderedDict
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras.utils import pad_sequences
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text
from keras_preprocessing.sequence import pad_sequences
from keras_contrib.metrics import crf_viterbi_accuracy
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


class PDFProcessor:
    def __init__(self, pdf_path, image_folder_path):
        self.pdf_path = pdf_path
        self.image_folder_path = image_folder_path
        self.pdf_images = None
        self.pages = None

    def convert_pdf_to_images(self):
        image_folder_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
        self.image_folder_path = os.path.join(self.image_folder_path, image_folder_name)

        # Create the image folder if it doesn't exist
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)

        # Convert the PDF file to JPEG images and save them to the image folder
        images = convert_from_path(self.pdf_path, dpi=300, fmt="jpg")

        for i, image in enumerate(images):
            # Construct the filename using the PDF filename and page number
            filename = f"{os.path.splitext(os.path.basename(self.pdf_path))[0]}_{i + 1}.jpg"
            image_path = os.path.join(self.image_folder_path, filename)
            # Save the image to the specified path
            image.save(image_path, "JPEG")

        self.pdf_images = image_folder_name
        self.pages = len(images)


class YOLOv5Detector:
    def __init__(self, weights_path, conf_threshold, source_path, yolo_folder):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.source_path = source_path
        self.yolo_folder = yolo_folder

    def run_detection(self):
        os.chdir(self.yolo_folder)
        cmd = f"python3 {self.yolo_folder}/detect.py --weights {self.weights_path} --conf {self.conf_threshold} --source {self.source_path} --save-txt"
        subprocess.run(cmd, shell=True)

    def get_annotations(self):
        parent_folder = os.path.join(self.yolo_folder, "runs", "detect")
        subfolders = os.listdir(parent_folder)
        subfolders = [subfolder for subfolder in subfolders if subfolder != "exp"]
        subfolder_numbers = [
            int(subfolder.split("exp")[-1])
            for subfolder in subfolders
            if "exp" in subfolder and subfolder.split("exp")[-1].isdigit()
        ]

        if subfolder_numbers:
            latest_subfolder_number = max(subfolder_numbers)
            latest_subfolder = f"exp{latest_subfolder_number}"
        else:
            latest_subfolder = "exp"

        annotation_folder = os.path.join(parent_folder, latest_subfolder, "labels")
        return annotation_folder


class DataExtractor:
    def __init__(self, pdf_images, annotation_folder):
        self.pdf_images = pdf_images
        self.annotation_folder = annotation_folder

    def extract_data_v5(self, pdf_images, annotation_folder):
        customer_name, vendor_name, cvr, invoice_no, due_date = "", "", "", "", ""
        product_number, content_unit, product_price = [], [], []

        img_paths = sorted(glob.glob(pdf_images + "/*.jpg"))

        for img_path in img_paths:
            # load the image
            img = cv2.imread(img_path)

            coords = "0 0.681648 0.091135 0.561917 0.090637"
            id, x, y, w, h = map(float, coords.split())

            l = int((x - w / 2) * img.shape[1])
            r = int((x + w / 2) * img.shape[1])
            t = int((y - h / 2) * img.shape[0])
            b = int((y + h / 2) * img.shape[0])

            roi = img[t:b, l:r]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            kernel = np.ones((3, 100), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            text = pytesseract.image_to_string(roi, lang="dan")

            if "inco København Cash & Carry A/S" in text:
                vendor_name = "inco København Cash & Carry A/S"

            # extract the file name from img_path
            file_name = os.path.basename(img_path)
            file_name = os.path.splitext(file_name)[0]

            # construct the annotation_path
            annotation_path = os.path.join(annotation_folder, file_name + ".txt")

            with open(annotation_path, "r") as f:
                data = f.readlines()

            sorted_data = sorted(
                data, key=lambda x: (float(x.split(" ")[0]), float(x.split(" ")[2]))
            )

            # Loop through the annotations and draw the boxes
            for dt in sorted_data:
                id, x, y, w, h = map(float, dt.split(" "))
                l = int((x - w / 2) * img.shape[1])
                r = int((x + w / 2) * img.shape[1])
                t = int((y - h / 2) * img.shape[0])
                b = int((y + h / 2) * img.shape[0])

                roi = img[t:b, l:r]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                kernel = np.ones((3, 100), np.uint8)
                dilated = cv2.dilate(thresh, kernel, iterations=1)

                text = pytesseract.image_to_string(roi, lang="dan")

                label = int(id)
                if label == 0:
                    text = text.replace("”", "*")
                    customer_name = text.strip()

                elif label == 1:
                    vendor_name = text.strip()

                elif label == 2:
                    # cvr = text.strip()
                    cvr_values = re.findall(r"\b\d+\b", text)
                    # Join the numeric values into a single string
                    cvr = "".join(cvr_values)

                elif label == 3:
                    invoice_no = text.strip()

                elif label == 4:
                    due_date = text.strip()

                elif label == 5:
                    product_number.extend(filter(None, text.strip().split("\n")))
                elif label == 6:
                    content_unit.extend(filter(None, text.strip().split("\n")))

                elif label == 7:
                    numeric_values = re.findall(r"\b\d+(?:,\s*\d+)*\b", text)
                    product_price.extend(numeric_values)

        print("product_number_len: ", len(product_number))
        print("product_price_len: ", len(product_price))
        return (
            customer_name,
            vendor_name,
            cvr,
            invoice_no,
            due_date,
            product_number,
            content_unit,
            product_price,
        )

    def extract_data_v7(self, pdf_images, annotation_folder):

        extracted_data = []
        invoice_date = ""
        quantity = []
        product_name = []
        product_total = []
        total_box = []
        total_excluding_tax = ""
        tax = ""
        total_including_tax = ""

        antal = []
        enhed = []

        isSpecial = False
        content_indexes = []

        sum = 0

        img_paths = sorted(glob.glob(pdf_images + "/*.jpg"))

        for img_path in img_paths:
            isOverfort = False

            # Read the image
            img = cv2.imread(img_path)

            # extract the file name from img_path
            file_name = os.path.basename(img_path)
            file_name = os.path.splitext(file_name)[0]

            # construct the annotation_path
            annotation_path = os.path.join(annotation_folder, file_name + ".txt")

            with open(annotation_path, "r") as f:
                data = f.readlines()

            sorted_data = sorted(
                data, key=lambda x: (float(x.split(" ")[0]), float(x.split(" ")[2]))
            )

            text = pytesseract.image_to_string(img, lang="dan")

            if "Ordreliniertotal" in text:
                isSpecial = True

            if "Overført" in text:
                isOverfort = True

            # Loop through the annotations and draw the boxes
            for dt in sorted_data:
                id, x, y, w, h = map(float, dt.split(" "))
                l = int((x - w / 2) * img.shape[1])
                r = int((x + w / 2) * img.shape[1])
                t = int((y - h / 2) * img.shape[0])
                b = int((y + h / 2) * img.shape[0])

                roi = img[t:b, l:r]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                kernel = np.ones((3, 100), np.uint8)
                dilated = cv2.dilate(thresh, kernel, iterations=1)

                text = pytesseract.image_to_string(
                    roi, config="--oem 3 --psm 6", lang="dan"
                )

                if id == 0:
                    text = " ".join(text.split("\n"))
                    invoice_date = text.strip()

                elif id == 1:
                    quantity.extend(
                        line.strip() for line in text.split("\n") if line.strip()
                    )

                elif id == 2:
                    lines = text.split("\n")
                    lines = [
                        line.strip()
                        for line in lines
                        if line
                           and not line.startswith("Subtotal")
                           and "Overført" not in line
                           and not line.startswith("D-mærke")
                    ]
                    product_name.extend(lines)

                elif id == 3:
                    # product_total.extend(line.strip() for line in text.split('\n') if line.strip())
                    numeric_values = re.findall(r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)\b", text)

                    product_total.extend(numeric_values)

                elif id == 4:
                    total_box.extend(
                        line.strip() for line in text.split("\n") if line.strip()
                    )

                    # Modify the elements in total_box to only contain the numeric values
                    for i, val in enumerate(total_box):
                        total_box[i] = val.split()[-1]

                    # Extract values from the modified total_box list
                    total_excluding_tax = total_box[0]
                    tax = total_box[1]
                    total_including_tax = total_box[2]

            for value in quantity:
                value = value.split(" ")
                antal.append(value[0])
                enhed.append(value[1])

        is_content_section = False

        product_name_cleaned = [line for line in product_name if not line.isupper()]

        if not isSpecial:
            for i, val in enumerate(product_name):
                if val == "GRØNT":
                    # If 'GRØNT' is found, we start looking for the next product and append its index
                    is_content_section = True
                    next_product_index = i + 1

                    # Check if the next product index is within the range of product_name
                    while next_product_index < len(product_name):
                        next_product = product_name[next_product_index]
                        if next_product in product_name_cleaned:
                            index_in_cleaned = product_name_cleaned.index(next_product)
                            content_indexes.append(index_in_cleaned)
                            next_product_index += 1
                        else:
                            break

                elif val.isupper() and is_content_section:
                    # If an all uppercase item is found and we were in a content section, we stop appending indexes
                    break

        # for prod in product_name:
        #     print(prod)

        # print('-------------------')

        # for prod in product_name_cleaned:
        #     print(prod)

        # print(content_indexes)

        extracted_data = (
            invoice_date,
            antal,
            enhed,
            product_name,
            product_total,
            total_excluding_tax,
            tax,
            total_including_tax,
            content_indexes,
        )

        print("product_name_len: ", len(product_name))
        print("product_total_len: ", len(product_total))
        print("antal_len: ", len(antal))
        print("enhed_len: ", len(enhed))

        return extracted_data

    def get_combined_data(self, extracted_data_v5, extracted_data_v7, pages, nlp_output):
        (
            customer_name,
            vendor_name,
            cvr,
            invoice_no,
            due_date,
            product_numbers,
            content_unit,
            product_price,
        ) = extracted_data_v5
        (
            invoice_date,
            antal,
            enhed,
            product_name,
            product_total,
            total_excluding_tax,
            tax,
            total_including_tax,
            _,
        ) = extracted_data_v7

        result = {
            "invoice_number": invoice_no,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "total_pages": pages,
            "sub_total": total_excluding_tax,
            "vat": tax,
            "total": total_including_tax,
            "vendor": {
                "name": vendor_name,
                "cvr": cvr,
                "address": None,
                "email": None,
            },
            "company": {
                "id": None,
                "name": customer_name,
                "cvr": cvr,
                "address": None,
                "email": None,
            },
            "products": [],
        }
        try:
            for i in range(len(product_numbers)):
                product = {
                    "product_number": product_numbers[i],
                    "product_desc": nlp_output[i],
                    "antal": antal[i],
                    "enhed": enhed[i],
                    "content_unit": content_unit[i],
                    "price": product_price[i],
                    "total": product_total[i],
                }
                result["products"].append(product)
        except:
            print("Error in creating products")
            pass

        return result


class NERModel:
    def __init__(self, model_path, word2idx_path, tag2idx_path):
        self.model_path = model_path
        self.word2idx_path = word2idx_path
        self.tag2idx_path = tag2idx_path

        self.max_len = 1000
        self.model = None
        self.word2idx = None
        self.tag2idx = None

    def load_model(self):
        self.model = load_model(
            self.model_path,
            custom_objects={"CRF": CRF, "crf_loss": crf_loss, "crf_viterbi_accuracy": crf_viterbi_accuracy},
        )

        with open(self.word2idx_path, "rb") as f:
            self.word2idx = joblib.load(f)

        with open(self.tag2idx_path, "rb") as f:
            self.tag2idx = joblib.load(f)

    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}

    def test_sentence_sample(test_sentence):
        results = []
        x_test_sent = pad_sequences(
            sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]],
            padding="post",
            value=0,
            maxlen=max_len,
        )
        p = model_predict.predict(np.array([x_test_sent[0]]))
        p = np.argmax(p, axis=-1)
        for w, pred in zip(tokenize(test_sentence), p[0]):
            results.append([w, tags[pred]])
        return results

    def tokenize(s):
        if isinstance(s, (list, tuple)):
            s = "\n".join(map(str, s))
        return s.split("\n")

    def remove_empty_lines(self, text):
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        return "\n".join(non_empty_lines)

    def process_text(self, text):
        lines = text.split("\n")
        processed_lines = []

        for line in lines:

            line = re.sub(r'\s+', ' ', line)

            if " f " in line:
                regex = r" f (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(line.replace(" f " + match.group(1), ""))
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)
            elif line.startswith("f "):
                regex = r"f (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append("")
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)

            elif re.match(r"^\s*ø (.+)", line):
                regex = r"ø (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(match.group(1))


            elif "fø " in line:
                regex = r"fø (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append("")
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)
            # Add more conditions here...
            elif re.search(r"\s(?:f|ø|fø)\s", line):
                regex = r"\s(?:f|ø|fø)\s(.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(line.replace(match.group(0), ""))
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)
            elif re.search(r"^\d+\s[A-Z]+\s", line):
                regex = r"^(\d+\s[A-Z]+\s)(.*)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(match.group(1))
                    processed_lines.append(match.group(2))
                else:
                    processed_lines.append(line)
            elif " ÆSK " in line:
                regex = r" ÆSK (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(line.replace(" ÆSK " + match.group(1), ""))
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)
            elif " SÆK " in line:
                regex = r" SÆK (.+)"
                match = re.search(regex, line)
                if match:
                    processed_lines.append(line.replace(" SÆK " + match.group(1), ""))
                    processed_lines.append(match.group(1))
                else:
                    processed_lines.append(line)
            # Skip lines starting with numeric value and space, or containing only uppercase text
            elif re.match(r"^\d+\s[A-ZÆØÅÉÆØÅa-z0-9\s\-!@#$%^&*()_+=[\]{}|;:'\",.<>?/\\]+", line):
                continue
            # Skip lines containing only numeric values with length less than 4
            elif line.isdigit() and len(line) < 4:
                continue
            # Skip lines starting with numeric value containing dot, comma, or both
            elif re.match(r"^-?\d+[,.]\d+", line):
                continue
            # Skip lines starting with "D-mærke"
            elif line.startswith("D-mærke"):
                continue
            elif line in ["D", "f", "ø", "fø"]:
                continue
            elif re.match(r"^-", line):
                continue
            elif "Subtotal" == line:
                continue
            elif re.match(r"^\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.\s*\.$", line):
                continue
            else:
                processed_lines.append(line)
        return processed_lines

    def extract_lines_with_condition(self, lines):
        processed_lines = []
        start_extraction = False

        for line in lines:
            if "Varenr." in line:
                start_extraction = True
            if start_extraction and "subtotal" in line:
                break
            if start_extraction and re.match(r"^-?\d+\s[A-Z]+", line):
                continue
            if re.match(r"^[A-ZÆØÅÉ]+$", line):
                continue
            if start_extraction:
                processed_lines.append(line)

        return processed_lines

    def extract_products_faktura(lines):
        processed_lines = []
        extract_next_line = False

        for line in lines:
            if not line.strip():  # Skip empty lines
                continue
            if re.match(r"^\d{4,6}$", line) and '.' not in line and ',' not in line:
                extract_next_line = True
            elif extract_next_line:
                processed_lines.append(line)
                extract_next_line = False
        return processed_lines

    def extract_products_kreditnota(lines):
        processed_lines = []
        previous_line = None

        for line in lines:
            if re.match(r"^\d{4,6}$", line) and '.' not in line and ',' not in line:
                if previous_line is not None:
                    processed_lines.append(previous_line)
                previous_line = None
            else:
                previous_line = line

            if not line.strip():  # Skip empty lines
                continue

            previous_line = line

        return processed_lines

    def split_text_into_lines(text):
        lines = text.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        return non_empty_lines

    def split_text(lines):
        return lines.split()

    def product_extraction(pdf_path):
        full_processed_text = ""
        with fitz.open(pdf_path) as pdf:
            for page_number in range(pdf.page_count):
                page = pdf.load_page(page_number)
                page_text = page.get_text()

                page_text_cleaned = page_text.replace("�", " ")
                processed_text = process_text(page_text_cleaned)
                trimmed_text = extract_lines_with_condition(processed_text)
                if "Kreditnota" not in page_text:
                    products_faktura = extract_products_faktura(trimmed_text)
                if "Kreditnota" in page_text:
                    products_kreditnota = extract_products_kreditnota(trimmed_text)

                products = products_kreditnota if "Kreditnota" in page_text else products_faktura

                processed_text_as_string = "\n".join(products)

                full_processed_text += processed_text_as_string + "\n"

        return full_processed_text

    def pdf_to_model_predictions(pdf_path):
        predictions = []
        with fitz.open(pdf_path) as pdf:
            for page_number in range(pdf.page_count):
                page = pdf.load_page(page_number)
                page_text = page.get_text()

                page_text_cleaned = page_text.replace("�", " ")
                text_tokens = split_text(page_text_cleaned)
                page_predictions = test_sentence_sample(text_tokens)
                predictions.extend(page_predictions)

        return predictions

    def post_processing(predictions):
        descriptions = []
        for text, tag in predictions:
            if tag in ["B-BESKRIVELSE", "I-BESKRIVELSE"]:
                descriptions.append(text)
        return descriptions

    def results_post_processing(data):
        descriptions = []
        buffer = []

        for entry in data:
            if entry[1] == 'B-BESKRIVELSE':
                if buffer:
                    descriptions.append(' '.join(buffer))
                    buffer = []
                buffer.append(entry[0])
            elif entry[1] == 'I-BESKRIVELSE':
                buffer.append(entry[0])
        if buffer:
            descriptions.append(' '.join(buffer))

        return descriptions


class PDFAPI:
    def __init__(self):
        self.pdf_processor = None
        self.yolov5_detector = None
        self.yolov7_detector = None
        self.data_extractor = None
        self.ner_model = None

    def process_pdf(self, pdf_path, image_folder_path):
        self.pdf_processor = PDFProcessor(pdf_path, image_folder_path)
        self.pdf_processor.convert_pdf_to_images()

    def run_yolo_detection(self, v5weights_path, v7weights_path, conf_threshold, yolo_folder):
        self.yolov5_detector = YOLOv5Detector(v5weights_path, conf_threshold, self.pdf_processor.pdf_images,
                                              yolo_folder)
        self.yolov7_detector = YOLOv5Detector(v7weights_path, conf_threshold, self.pdf_processor.pdf_images,
                                              yolo_folder)

        self.yolov5_detector.run_detection()
        self.yolov7_detector.run_detection()

    def extract_data(self, v5annotation_folder, v7annotation_folder):
        self.data_extractor = DataExtractor(self.pdf_processor.pdf_images, v7annotation_folder)
        extracted_data_v5 = self.data_extractor.extract_data_v5()
        extracted_data_v7 = self.data_extractor.extract_data_v7()
        combined_data = self.data_extractor.get_combined_data()
        return combined_data

    def load_ner_model(self, model_path, word2idx_path, tag2idx_path):
        self.ner_model = NERModel(model_path, word2idx_path, tag2idx_path)
        self.ner_model.load_model()


API_KEY = "7cd12d16-286f-4b6f-a42b-7d55557e5b65"  # Replace with your actual API key

api_key_header = APIKeyHeader(name="NER_API_Key")


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return True
    else:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.post("/predict/inco")
async def main(
        files: UploadFile = File(..., media_type="application/pdf"),
        authorized: bool = Depends(verify_api_key),
):
    # Process PDF
    pdf_path = "/tmp/uploaded.pdf"  # Specify a temporary location for the uploaded PDF
    image_folder_path = "/tmp/images"  # Specify a temporary location for storing images
    pdf_api = PDFAPI()
    pdf_api.process_pdf(pdf_path, image_folder_path)

    # Run YOLO Detection
    v5weights_path = "/path/to/v5/weights.pt"
    v7weights_path = "/path/to/v7/weights.pt"
    conf_threshold = 0.3
    yolo_folder = "/path/to/yolov5"
    pdf_api.run_yolo_detection(v5weights_path, v7weights_path, conf_threshold, yolo_folder)

    # Extract Data
    v5annotation_folder = pdf_api.yolov5_detector.get_annotations()
    v7annotation_folder = pdf_api.yolov7_detector.get_annotations()
    combined_data = pdf_api.extract_data(v5annotation_folder, v7annotation_folder)

    # Load NER Model
    ner_model_path = "/path/to/ner/model.h5"
    word2idx_path = "/path/to/word2idx.pkl"
    tag2idx_path = "/path/to/tag2idx.pkl"
    pdf_api.load_ner_model(ner_model_path, word2idx_path, tag2idx_path)

    # Test Sentence Sample
    test_sentence = "Your test sentence here."
    sentence_data = pdf_api.ner_model.test_sentence_sample(test_sentence)

    return {"combined_data": combined_data, "sentence_data": sentence_data}
