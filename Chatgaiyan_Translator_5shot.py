import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

def append_csv_line(csv_path, row):
    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def translate_bangla_to_chatgaiyan(row):
    if len(row) < 2:
        raise ValueError("Row must have at least two elements: id and bangla_text")
    id_val = row[0]
    bangla_text = row[1]
    prompt = f"""You are a precise translation tool. Your only task is to translate the given Bangla text to Chittagonian (Chatgaiyan) dialect using Bengali script.

    INSTRUCTIONS:
    - Translate the Bangla text below to Chittagonian (Chatgaiyan) dialect
    - Use only Bengali script (not Latin script or IPA)
    - Return ONLY the translated text with no additional commentary, explanations, or notes
    - Do not include phrases like "Here is the translation:" or "The Chittagonian translation is:"
    - Do not add any metadata, formatting, or extra information
    - If you cannot translate a specific word, keep it as is in the original form

    Here are some examples of Bangla to Chittagonian translations:
    Bangla: খালু, গ্রামের বাড়ি যাবেন কবে
    Chittagonian: অহালু, গ্রামর বাড়িত হত্তে যাইবেন?

    Bangla: আপনি কি নিয়মিত জল খান?
    Chittagonian: অনে কি পইত্তেন পানি হন না?

    Bangla: জুম্মার পর যাব।
    Chittagonian: জুমর পর যাইয়ুম।

    Bangla: পালং শাক খুব পছন্দ
    Chittagonian: পালংশাক বেশি গম লাগে দে

    Bangla: বাঙালি নানা-নানীরা কি করেন?
    Chittagonian: বাআলর নানানানী অলে কি গরে দে?

    Bangla text to translate:
    {bangla_text}

Chittagonian translation:"""    
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        chatgaiyan_text = response.text.strip()
    except Exception as e:
        chatgaiyan_text = f"Translation error: {e}"
    return [id_val, chatgaiyan_text]

def extract_after_first_backslash(path):
    idx = path.find('\\')
    if idx != -1:
        return path[idx+1:]
    return path

def generate_chatgaiyan(csv_path, num_lines=None):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        count = 0
        for row in reader:
            path = extract_after_first_backslash(csv_path)
            append_csv_line('Translated_Chatgaiyan_5_Shot/' + path, translate_bangla_to_chatgaiyan(row))
            count += 1
            if num_lines is not None and count >= num_lines:
                break

if __name__ == "__main__":
    generate_chatgaiyan(r'BangaCHQ\train.csv', num_lines=5)