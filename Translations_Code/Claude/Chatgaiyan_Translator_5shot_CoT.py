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
    prompt = f"""You are a precise translation tool. Your task is to translate the given Bangla text to Chittagonian (Chatgaiyan) dialect using Bengali script.

        INSTRUCTIONS:
        - Analyze the text step-by-step before translating
        - Use only Bengali script in your final translation
        - After your reasoning, provide ONLY the final translation with no additional commentary

        Here are examples showing the translation process:

        Example 1:
        Bangla: খালু, গ্রামের বাড়ি যাবেন কবে
        Reasoning: "খালু" becomes "অহালু" (phonetic shift with অ prefix), "গ্রামের বাড়ি" becomes "গ্রামর বাড়িত" (possessive -এর→-র, locative case -ত added), "যাবেন কবে" becomes "হত্তে যাইবেন?" (emphatic হত্তে added, verb form changed, question marker)
        Chittagonian: অহালু, গ্রামর বাড়িত হত্তে যাইবেন?

        Example 2:
        Bangla: আপনি কি নিয়মিত জল খান?
        Reasoning: "আপনি" becomes "অনে" (phonetic compression), "কি নিয়মিত" becomes "কি পইত্তেন" (restructured with পইত্তেন for regularity concept), "জল খান" becomes "পানি হন না?" (জল→পানি, খান→হন না with negative question form)
        Chittagonian: অনে কি পইত্তেন পানি হন না?

        Example 3:
        Bangla: জুম্মার পর যাব।
        Reasoning: "জুম্মার" becomes "জুমর" (possessive -আর→-র shortened), "পর" stays as "পর" (unchanged), "যাব" becomes "যাইয়ুম" (first person future tense -ব→-ইয়ুম)
        Chittagonian: জুমর পর যাইয়ুম।

        Example 4:
        Bangla: পালং শাক খুব পছন্দ
        Reasoning: "পালং শাক" stays as "পালংশাক" (compound unchanged), "খুব" becomes "বেশি" (lexical replacement), "পছন্দ" becomes "গম লাগে দে" (idiomatic expression with গম=liking, লাগে=feels, দে=emphasis particle)
        Chittagonian: পালংশাক বেশি গম লাগে দে

        Example 5:
        Bangla: বাঙালি নানা-নানীরা কি করেন?
        Reasoning: "বাঙালি" becomes "বাআলর" (phonetic compression with possessive -র), "নানা-নানী-রা" becomes "নানানানী অলে" (compound simplified, plural -রা→অলে), "কি করেন" becomes "কি গরে দে?" (করেন→গরে with দে emphasis particle)
        Chittagonian: বাআলর নানানানী অলে কি গরে দে?

        Now translate this text:
        Bangla: {bangla_text}

        Think through the translation:
        - Identify key vocabulary transformations
        - Note grammatical changes (possessives, verb forms, cases)
        - Apply Chittagonian phonetic and structural patterns

        Provide your final translation below:
        Chittagonian:"""      
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
            append_csv_line('Translation_Output\\Claude\\Translated_Chatgaiyan_5_Shot\\' + path, translate_bangla_to_chatgaiyan(row))
            count += 1
            if num_lines is not None and count >= num_lines:
                break

if __name__ == "__main__":
    generate_chatgaiyan(r'Split_Dataset\train3.csv')