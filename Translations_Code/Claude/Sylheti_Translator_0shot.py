import os
from dotenv import load_dotenv
import anthropic
import csv

load_dotenv()

API_KEY = os.getenv("CLAUDE_API_KEY")

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
    prompt = f"""You are a precise translation tool. Your only task is to translate the given Bangla text to Sylheti dialect using Bengali script.

    INSTRUCTIONS:
    - Translate the Bangla text below to Sylheti dialect
    - Use only Bengali script (not Latin script or IPA)
    - Return ONLY the translated text with no additional commentary, explanations, or notes
    - Do not include phrases like "Here is the translation:" or "The Sylheti translation is:"
    - Do not add any metadata, formatting, or extra information
    - If you cannot translate a specific word, keep it as is in the original form

    Bangla text to translate:
    {bangla_text}

Sylheti translation:"""    
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        chatgaiyan_text = message.content[0].text.strip()
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
            append_csv_line('Translation_Output\\Claude\\Translated_Chatgaiyan_0_Shot\\' + path, translate_bangla_to_chatgaiyan(row))
            count += 1
            if num_lines is not None and count >= num_lines:
                break

if __name__ == "__main__":
    generate_chatgaiyan(r'Split_Dataset\train3.csv')