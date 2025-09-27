import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY_5")

def append_csv_line(csv_path, row):
    dir_name = os.path.dirname(csv_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def translate_bangla_to_sylheti(row):
    if len(row) < 2:
        raise ValueError("Row must have at least two elements: id and bangla_text")
    id_val = row[0]
    bangla_text = row[1]
    prompt = f"""You are a precise translation tool. Your only task is to translate the given Bangla text to Sylheti dialect using Bengali script.

    INSTRUCTIONS:
    - Translate the Bangla text below to Sylheti dialect
    - Use only Bengali script (not Latin script or IPA)
    - Return ONLY the translated text with no additional commentary, explanations, or notes
    - Do not include phrases like "Here is the translation:" or "The Sylheti translation is:"
    - Do not add any metadata, formatting, or extra information
    - If you cannot translate a specific word, keep it as is in the original form

    Here are some examples of Bangla to Sylheti translations:
    Bangla: আপনি কি নিয়মিত স্ক্যান করেন?
    Sylheti: আফনে কিতা রেগুলার স্ক্যান খরইননি?

    Bangla: কোথায় কোচিং করেন?
    Sylheti: কোনানো কোচিং খরইন?

    Bangla: ডাল সিদ্ধ হয়েছে?
    Sylheti: ডাইল সিদ্ধ অইছেনি?

    Bangla: এই যে মিয়া, কি নিয়ে যাও?
    Sylheti: ওবা, কিতা লইয়া যাইরায়?

    Bangla: আপনি কি নিয়মিত স্ক্যান করেন?
    Sylheti: আফনে কিতা রেগুলার স্ক্যান খরইননি?

    Bangla: হ্যাঁ, প্রতিদিন করি
    Sylheti: অয়, ডেইলি খরি

    Bangla: কি করছেন?
    Sylheti: কিতা খররা?

    Bangla: জামাই কবে আসবে?
    Sylheti: জামাই কোনদিন আইবো?

    Bangla: হলুদ আর মরিচ দেন।
    Sylheti: অলইদ আর মরিচ দেইন।

    Bangla: দুটো দিলেই হবে।
    Sylheti: দুইটা দিলে ওউ অইবো।

    Bangla text to translate:
    {bangla_text}

Sylheti translation:"""    
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        sylheti_text = response.text.strip()
    except Exception as e:
        sylheti_text = f"Translation error: {e}"
    return [id_val, sylheti_text]

def extract_after_first_backslash(path):
    idx = path.find('\\')
    if idx != -1:
        return path[idx+1:]
    return path

def generate_sylhetti(csv_path, num_lines=None):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        count = 0
        for row in reader:
            path = extract_after_first_backslash(csv_path)
            append_csv_line(r'Translated_Sylhet_10_Shot\\' + path, translate_bangla_to_sylheti(row))
            count += 1
            if num_lines is not None and count >= num_lines:
                break


if __name__ == "__main__":
    # test_gemini_api()
    generate_sylhetti(r'Split_Dataset\train3.csv')
    