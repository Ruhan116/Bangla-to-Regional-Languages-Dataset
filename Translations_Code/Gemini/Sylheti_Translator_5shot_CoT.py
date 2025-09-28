import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY_4")

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
    prompt = f"""You are a precise translation tool. Your task is to translate the given Bangla text to Sylheti dialect using Bengali script.

        INSTRUCTIONS:
        - Analyze the text step-by-step before translating
        - Use only Bengali script in your final translation
        - After your reasoning, provide ONLY the final translation with no additional commentary

        Here are examples showing the translation process:

        Example 1:
        Bangla: আপনি কি নিয়মিত স্ক্যান করেন?
        Reasoning: "আপনি" becomes "আফনে" (phonetic shift প→ফ, ই→ে), "কি নিয়মিত" becomes "কিতা রেগুলার" (কি→কিতা with emphasis particle, নিয়মিত→রেগুলার lexical change), "স্ক্যান করেন" becomes "স্ক্যান খরইননি?" (করেন→খরইননি with খ sound and -ননি ending for question)
        Sylheti: আফনে কিতা রেগুলার স্ক্যান খরইননি?

        Example 2:
        Bangla: কোথায় কোচিং করেন?
        Reasoning: "কোথায়" becomes "কোনানো" (locative form কোথায়→কোনানো with extended vowel), "কোচিং" stays as "কোচিং" (unchanged), "করেন" becomes "খরইন?" (verb করেন→খরইন with খ sound and -ইন ending)
        Sylheti: কোনানো কোচিং খরইন?

        Example 3:
        Bangla: ডাল সিদ্ধ হয়েছে?
        Reasoning: "ডাল" becomes "ডাইল" (vowel shift আ→আই for diphthong), "সিদ্ধ" stays as "সিদ্ধ" (unchanged), "হয়েছে" becomes "অইছেনি?" (হয়েছে→অইছেনি with অ prefix and -নি question marker)
        Sylheti: ডাইল সিদ্ধ অইছেনি?

        Example 4:
        Bangla: এই যে মিয়া, কি নিয়ে যাও?
        Reasoning: "এই যে মিয়া" becomes "ওবা" (colloquial address form, complete lexical replacement for calling attention), "কি নিয়ে" becomes "কিতা লইয়া" (কি→কিতা, নিয়ে→লইয়া with ল sound), "যাও" becomes "যাইরায়?" (যাও→যাইরায় with extended verb form and য় ending)
        Sylheti: ওবা, কিতা লইয়া যাইরায়?

        Example 5:
        Bangla: জুম্মার পর যাব।
        Reasoning: "জুম্মার" becomes "জুম্মার" (unchanged as Arabic loanword), "পর" stays as "পর" (unchanged), "যাব" becomes "যামু" (first person future -ব→-মু typical Sylheti verb ending)
        Sylheti: জুম্মার পর যামু।

        Now translate this text:
        Bangla: {bangla_text}

        Think through the translation:
        - Identify key vocabulary transformations
        - Note grammatical changes (verb forms, pronouns, question markers)
        - Apply Sylheti phonetic patterns (খ for ক in verbs, diphthongs, vowel shifts)

        Provide your final translation below:
        Sylheti:"""    
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
            append_csv_line(r'Translated_Sylhet_5_Shot\\' + path, translate_bangla_to_sylheti(row))
            count += 1
            if num_lines is not None and count >= num_lines:
                break


if __name__ == "__main__":
    # test_gemini_api()
    generate_sylhetti(r'Split_Dataset\train3.csv')
    