from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form, Depends
from typing import Optional, Dict
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import io

from strsimpy.normalized_levenshtein import NormalizedLevenshtein

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

import cv2
import pytesseract



app = FastAPI()

# Load the parquet file containing medicine names
FILE = "./medicines.parquet"  
medicines = pd.read_parquet(FILE)

medicine_names = medicines["name"].tolist()


def findSimilarity(w1, w2):
    normalized_levenshtein = NormalizedLevenshtein()

    return normalized_levenshtein.similarity(w1, w2)

def extract_text_from_cv2(image_bytes):
    # Load image
    # image = cv2.imread(path)
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract text
    extracted_text = pytesseract.image_to_string(gray)
    
    return extracted_text.strip()


# @app.get("/match-medicine/")
def match_medicine(query: str):
    """
    Finds the top 3 similar medicine names based on the input query.
    """

    all_simi = pd.DataFrame(columns=['input','match','score'])
    for idx, row in medicines.iterrows():
        w1 = query
        for col in ['name','generic_name']:
            w2 = row[col]
            simi = findSimilarity(w1, w2)
            all_simi.loc[len(all_simi)] = [w1, w2, simi]
    matches = all_simi.sort_values('score',ascending=False).head(3)[['match','score']].values
    return [
        {"medicine": match[0], "similarity_score": np.round(match[1],3)}
        for match in matches
    ]

# def get_form_data(
#     query: Optional[str] = Form(None),
#     file: Optional[UploadFile] = File(None)
# ) -> Dict[str, Optional[str]]:
#     """
#     Extracts query or file from the request.
#     Ensures at least one input (query or image) is provided.
#     """
#     if not query and not file:
#         raise HTTPException(status_code=400, detail="Either text query or image file is required")
    
#     return {"query": query, "file": file}


# async def match_text_or_image(query: Optional[str] = Form(None, description="Medicine name to match"),
#     file: Optional[UploadFile] = File(None)):
# async def match_text_or_image(query: str):
# async def match_text_or_image(file: Optional[UploadFile] = File(None),
#                               query: Optional[str] = Form(None, Description="medicine name to match")):

@app.post("/match-text/")
async def match_text(
    query: str
):
    """
    Accepts a text query and finds the closest matching medicines.
    """


    # Find top matching medicines
    matches = match_medicine(query)
    return {"input_text": query, "matches": matches}
    
@app.post("/match-image/")
async def match_image(
    file: UploadFile = File(...)
):
    """
    Accepts either a text query or an image, extracts text if needed, 
    and finds the closest matching medicines.
    """
    # Read image as bytes
    image_bytes = await file.read()
    
    # Perform OCR
    query = extract_text_from_cv2(image_bytes)

    # Find top matching medicines
    matches = match_medicine(query)
    return {"transcribed_text": query, "matches": matches}
