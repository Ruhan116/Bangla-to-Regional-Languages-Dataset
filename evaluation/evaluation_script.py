import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk import word_tokenize, ngrams
from collections import Counter
import nltk
from difflib import SequenceMatcher
from tqdm import tqdm
import os

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    ref = list(reference.replace(" ", ""))
    hyp = list(hypothesis.replace(" ", ""))
    
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return (d[len(ref)][len(hyp)] / len(ref)) * 100

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref = reference.split()
    hyp = hypothesis.split()
    
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return (d[len(ref)][len(hyp)] / len(ref)) * 100

def calculate_bleu(reference, hypothesis):
    """Calculate BLEU score"""
    ref_tokens = [reference.split()]
    hyp_tokens = hypothesis.split()
    
    smoothing = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothing) * 100

def calculate_bleu_ngrams(reference, hypothesis):
    """Calculate individual BLEU n-gram scores"""
    ref_tokens = [reference.split()]
    hyp_tokens = hypothesis.split()
    smoothing = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing) * 100
    bleu_2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing) * 100
    bleu_3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing) * 100
    bleu_4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing) * 100
    
    return bleu_1, bleu_2, bleu_3, bleu_4

def calculate_meteor(reference, hypothesis):
    """Calculate METEOR score"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    try:
        return meteor_score([ref_tokens], hyp_tokens) * 100
    except:
        return 0.0

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    
    return {
        'ROUGE-1': scores['rouge1'].fmeasure * 100,
        'ROUGE-2': scores['rouge2'].fmeasure * 100,
        'ROUGE-L': scores['rougeL'].fmeasure * 100
    }

def calculate_ter(reference, hypothesis):
    """Calculate Translation Edit Rate (TER)"""
    ref = reference.split()
    hyp = hypothesis.split()
    
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return (d[len(ref)][len(hyp)] / len(ref)) * 100

def calculate_length_ratio(reference, hypothesis):
    """Calculate length ratio between hypothesis and reference"""
    ref_len = len(reference.split())
    hyp_len = len(hypothesis.split())
    
    if ref_len == 0:
        return 0
    
    return (hyp_len / ref_len) * 100

def calculate_chrf(reference, hypothesis, n=6):
    """Calculate chrF score (character n-gram F-score)"""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    
    precision_sum = 0
    recall_sum = 0
    
    for i in range(1, n + 1):
        ref_ngrams = Counter([tuple(ref_chars[j:j+i]) for j in range(len(ref_chars)-i+1)])
        hyp_ngrams = Counter([tuple(hyp_chars[j:j+i]) for j in range(len(hyp_chars)-i+1)])
        
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        
        if sum(hyp_ngrams.values()) > 0:
            precision_sum += overlap / sum(hyp_ngrams.values())
        if sum(ref_ngrams.values()) > 0:
            recall_sum += overlap / sum(ref_ngrams.values())
    
    precision = precision_sum / n if n > 0 else 0
    recall = recall_sum / n if n > 0 else 0
    
    if precision + recall == 0:
        return 0
    
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score * 100

def calculate_jaccard_similarity(reference, hypothesis):
    """Calculate Jaccard similarity (word-level)"""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    
    intersection = len(ref_words & hyp_words)
    union = len(ref_words | hyp_words)
    
    if union == 0:
        return 0
    
    return (intersection / union) * 100

def calculate_cosine_similarity(reference, hypothesis):
    """Calculate cosine similarity (word-level)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    ref_vec = Counter(ref_words)
    hyp_vec = Counter(hyp_words)
    
    all_words = set(ref_vec.keys()) | set(hyp_vec.keys())
    
    dot_product = sum(ref_vec[word] * hyp_vec[word] for word in all_words)
    ref_magnitude = np.sqrt(sum(count ** 2 for count in ref_vec.values()))
    hyp_magnitude = np.sqrt(sum(count ** 2 for count in hyp_vec.values()))
    
    if ref_magnitude * hyp_magnitude == 0:
        return 0
    
    return (dot_product / (ref_magnitude * hyp_magnitude)) * 100

def calculate_levenshtein_similarity(reference, hypothesis):
    """Calculate normalized Levenshtein similarity"""
    ratio = SequenceMatcher(None, reference, hypothesis).ratio()
    return ratio * 100

def calculate_precision_recall_f1(reference, hypothesis):
    """Calculate word-level precision, recall, and F1"""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())
    
    if len(hyp_words) == 0:
        precision = 0
    else:
        precision = (len(ref_words & hyp_words) / len(hyp_words)) * 100
    
    if len(ref_words) == 0:
        recall = 0
    else:
        recall = (len(ref_words & hyp_words) / len(ref_words)) * 100
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1

def get_output_prefix(csv_path):
    """Extract first three words from filename for output naming"""
    filename = os.path.basename(csv_path)
    # Remove .csv extension
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    prefix = '_'.join(parts[:3]) if len(parts) >= 3 else name_without_ext
    return prefix , name_without_ext

def process_translations(human_csv_path, gemini_csv_path, text_column_index=1):
    """
    Process CSV files and calculate all metrics
    
    Parameters:
    -----------
    human_csv_path : str
        Path to CSV with HUMAN translations (REFERENCE/GROUND TRUTH)
    gemini_csv_path : str
        Path to CSV with GEMINI/LLM translations (HYPOTHESIS to evaluate)
    text_column_index : int
        Column index for translated text
    """
    
    # Read CSV files
    human_df = pd.read_csv(human_csv_path, header=None)
    gemini_df = pd.read_csv(gemini_csv_path, header=None)
    
    # Extract translation columns
    # HUMAN = REFERENCE (ground truth)
    reference_translations = human_df.iloc[:, text_column_index].tolist()
    # GEMINI = HYPOTHESIS (being evaluated)
    hypothesis_translations = gemini_df.iloc[:, text_column_index].tolist()
    
    # Initialize results storage
    results = []
    
    # Calculate metrics for each pair with progress bar
    print(f"\nProcessing {len(reference_translations)} translation pairs...")
    print("Evaluating Gemini/LLM translations against Human reference translations...\n")
    for idx in tqdm(range(len(reference_translations)), 
                    desc="Calculating metrics",
                    unit="pair"):
        reference = reference_translations[idx]
        hypothesis = hypothesis_translations[idx]
        # Skip if either text is NaN
        if pd.isna(reference) or pd.isna(hypothesis):
            continue
        
        reference = str(reference).strip()
        hypothesis = str(hypothesis).strip()
        
        # Calculate all metrics (reference = human, hypothesis = gemini)
        cer = calculate_cer(reference, hypothesis)
        wer = calculate_wer(reference, hypothesis)
        bleu = calculate_bleu(reference, hypothesis)
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_ngrams(reference, hypothesis)
        meteor = calculate_meteor(reference, hypothesis)
        rouge = calculate_rouge(reference, hypothesis)
        ter = calculate_ter(reference, hypothesis)
        length_ratio = calculate_length_ratio(reference, hypothesis)
        chrf = calculate_chrf(reference, hypothesis)
        jaccard = calculate_jaccard_similarity(reference, hypothesis)
        cosine = calculate_cosine_similarity(reference, hypothesis)
        levenshtein = calculate_levenshtein_similarity(reference, hypothesis)
        precision, recall, f1 = calculate_precision_recall_f1(reference, hypothesis)
        
        results.append({
            'Index': idx,
            'CER': cer,
            'WER': wer,
            'BLEU': bleu,
            'BLEU-1': bleu_1,
            'BLEU-2': bleu_2,
            'BLEU-3': bleu_3,
            'BLEU-4': bleu_4,
            'METEOR': meteor,
            'ROUGE-1': rouge['ROUGE-1'],
            'ROUGE-2': rouge['ROUGE-2'],
            'ROUGE-L': rouge['ROUGE-L'],
            'TER': ter,
            'chrF': chrf,
            'Jaccard': jaccard,
            'Cosine_Sim': cosine,
            'Levenshtein_Sim': levenshtein,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Length_Ratio': length_ratio
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    metrics = ['CER', 'WER', 'BLEU', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
               'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'TER', 'chrF', 
               'Jaccard', 'Cosine_Sim', 'Levenshtein_Sim', 'Precision', 'Recall', 
               'F1-Score', 'Length_Ratio']
    
    stats_data = []
    for metric in metrics:
        stats_data.append({
            'Metric': metric,
            'Mean': results_df[metric].mean(),
            'Std': results_df[metric].std(),
            'Min': results_df[metric].min(),
            'Max': results_df[metric].max(),
            'Median': results_df[metric].median()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Get output prefix from gemini CSV filename
    output_folder, output_prefix = get_output_prefix(gemini_csv_path)
    
    return results_df, stats_df, output_folder, output_prefix

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    human_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\evaluation\\translation-csv\\chatgaiyan_0_shot_human.csv"
    gemini_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\evaluation\\translation-csv\\chatgaiyan_0_shot_gemini.csv"
    
    # Process translations
    detailed_results, statistics, output_folder, output_prefix = process_translations(human_csv, gemini_csv, text_column_index=1)

    # Ensure output directory exists
    os.makedirs(f"evaluation\\results\\{output_folder}", exist_ok=True)
    
    # Create output filenames
    detailed_output = f"evaluation\\results\\{output_folder}\\{output_prefix}_detailed_metrics.csv"
    stats_output = f"evaluation\\results\\{output_folder}\\{output_prefix}_statistics.csv"
    
    # Display results
    print("\n" + "=" * 100)
    print("DETAILED RESULTS FOR EACH TRANSLATION PAIR")
    print("=" * 100)
    print(detailed_results.to_string(index=False))
    
    print("\n" + "=" * 100)
    print("STATISTICAL SUMMARY OF ALL METRICS")
    print("=" * 100)
    print(statistics.to_string(index=False))
    
    # Save results to CSV
    detailed_results.to_csv(detailed_output, index=False)
    statistics.to_csv(stats_output, index=False)
    
    print(f"\n✓ Results saved to '{detailed_output}' and '{stats_output}'")
    
    # Interpretation guide
    print("\n" + "=" * 100)
    print("METRIC INTERPRETATION GUIDE")
    print("=" * 100)
    print("ERROR RATES (Lower is better, 0 = perfect):")
    print("  • CER, WER, TER")
    print("\nSIMILARITY SCORES (Higher is better, 100 = perfect):")
    print("  • BLEU (1-4), METEOR, ROUGE, chrF")
    print("  • Jaccard, Cosine, Levenshtein")
    print("  • Precision, Recall, F1-Score")
    print("\nOTHER METRICS:")
    print("  • Length Ratio: 100 = same length, <100 = shorter, >100 = longer")
    print("=" * 100)