import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
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
    """Calculate Character Error Rate (lower is better, 0 = perfect match)"""
    if len(reference) == 0:
        return 100.0 if len(hypothesis) > 0 else 0.0
    
    ref = list(reference)
    hyp = list(hypothesis)
    
    # Levenshtein distance
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
    """Calculate Word Error Rate (lower is better, 0 = perfect match)"""
    ref = reference.split()
    hyp = hypothesis.split()
    
    if len(ref) == 0:
        return 100.0 if len(hyp) > 0 else 0.0
    
    # Levenshtein distance at word level
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
    """Calculate BLEU score (higher is better, 100 = perfect match)"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    # Using cumulative 4-gram BLEU (standard BLEU-4)
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return score * 100

def calculate_meteor(reference, hypothesis):
    """Calculate METEOR score (higher is better, 100 = perfect match)"""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    try:
        score = meteor_score([ref_tokens], hyp_tokens)
        return score * 100
    except:
        return 0.0

def calculate_rouge(reference, hypothesis):
    """Calculate ROUGE scores manually (higher is better, 100 = perfect match)"""
    if not reference.strip() or not hypothesis.strip():
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    # Tokenize by splitting on whitespace (works better for Bangla)
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    # ROUGE-1: Unigram overlap
    ref_unigrams = set(ref_tokens)
    hyp_unigrams = set(hyp_tokens)
    overlap_1 = len(ref_unigrams & hyp_unigrams)
    
    if len(ref_unigrams) == 0:
        rouge_1_recall = 0
    else:
        rouge_1_recall = overlap_1 / len(ref_unigrams)
    
    if len(hyp_unigrams) == 0:
        rouge_1_precision = 0
    else:
        rouge_1_precision = overlap_1 / len(hyp_unigrams)
    
    if rouge_1_recall + rouge_1_precision == 0:
        rouge_1 = 0
    else:
        rouge_1 = 2 * (rouge_1_precision * rouge_1_recall) / (rouge_1_precision + rouge_1_recall)
    
    # ROUGE-2: Bigram overlap
    ref_bigrams = [' '.join(ref_tokens[i:i+2]) for i in range(len(ref_tokens)-1)]
    hyp_bigrams = [' '.join(hyp_tokens[i:i+2]) for i in range(len(hyp_tokens)-1)]
    
    if len(ref_bigrams) == 0 or len(hyp_bigrams) == 0:
        rouge_2 = 0
    else:
        ref_bigrams_set = set(ref_bigrams)
        hyp_bigrams_set = set(hyp_bigrams)
        overlap_2 = len(ref_bigrams_set & hyp_bigrams_set)
        
        rouge_2_recall = overlap_2 / len(ref_bigrams) if len(ref_bigrams) > 0 else 0
        rouge_2_precision = overlap_2 / len(hyp_bigrams) if len(hyp_bigrams) > 0 else 0
        
        if rouge_2_recall + rouge_2_precision == 0:
            rouge_2 = 0
        else:
            rouge_2 = 2 * (rouge_2_precision * rouge_2_recall) / (rouge_2_precision + rouge_2_recall)
    
    # ROUGE-L: Longest Common Subsequence
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(ref_tokens, hyp_tokens)
    
    rouge_l_recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
    rouge_l_precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    
    if rouge_l_recall + rouge_l_precision == 0:
        rouge_l = 0
    else:
        rouge_l = 2 * (rouge_l_precision * rouge_l_recall) / (rouge_l_precision + rouge_l_recall)
    
    return {
        'ROUGE-1': rouge_1 * 100,
        'ROUGE-2': rouge_2 * 100,
        'ROUGE-L': rouge_l * 100
    }

def get_output_prefix(csv_path):
    """Extract first three words from filename for output naming"""
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    prefix = '_'.join(parts[:3]) if len(parts) >= 3 else name_without_ext
    return prefix, name_without_ext

def process_translations(human_csv_path, gemini_csv_path, text_column_index=1, id_column_index=0, max_rows=None):
    """
    Process CSV files and calculate translation metrics
    
    Parameters:
    -----------
    human_csv_path : str
        Path to CSV with HUMAN translations (REFERENCE/GROUND TRUTH)
    gemini_csv_path : str
        Path to CSV with GEMINI/LLM translations (HYPOTHESIS to evaluate)
    text_column_index : int
        Column index for translated text (default: 1)
    id_column_index : int
        Column index for ID/identifier (default: 0)
    max_rows : int or None
        Maximum number of rows to evaluate. If None, evaluates all matching rows
    """
    
    # Read CSV files
    human_df = pd.read_csv(human_csv_path, header=None)
    gemini_df = pd.read_csv(gemini_csv_path, header=None)
    
    # Extract ID and text columns
    human_df = human_df.rename(columns={id_column_index: 'id', text_column_index: 'text'})
    gemini_df = gemini_df.rename(columns={id_column_index: 'id', text_column_index: 'text'})
    
    print(f"\n{'='*80}")
    print(f"Human CSV: {len(human_df)} rows")
    print(f"Gemini CSV: {len(gemini_df)} rows")
    
    # Merge dataframes on ID to get only matching rows
    merged_df = pd.merge(
        human_df[['id', 'text']], 
        gemini_df[['id', 'text']], 
        on='id', 
        suffixes=('_human', '_gemini'),
        how='inner'
    )
    
    total_matches = len(merged_df)
    
    # Report matching statistics
    human_unmatched = len(human_df) - total_matches
    gemini_unmatched = len(gemini_df) - total_matches
    
    print(f"Matching IDs found: {total_matches}")
    if human_unmatched > 0:
        print(f"⚠ Human CSV has {human_unmatched} rows with IDs not found in Gemini CSV")
    if gemini_unmatched > 0:
        print(f"⚠ Gemini CSV has {gemini_unmatched} rows with IDs not found in Human CSV")
    
    # Apply max_rows limit if specified
    if max_rows is not None and max_rows < total_matches:
        merged_df = merged_df.head(max_rows)
        print(f"⚠ Limiting evaluation to first {max_rows} rows (as requested)")
    
    rows_to_process = len(merged_df)
    print(f"Rows to evaluate: {rows_to_process}")
    print(f"{'='*80}\n")
    
    # Initialize results storage
    results = []
    skipped_rows = []
    
    # Calculate metrics for each pair with progress bar
    print("Evaluating Gemini/LLM translations against Human reference translations...\n")
    
    for idx, row in tqdm(merged_df.iterrows(), 
                         total=rows_to_process,
                         desc="Calculating metrics",
                         unit="pair"):
        row_id = row['id']
        reference = row['text_human']
        hypothesis = row['text_gemini']
        
        # Skip if either text is NaN or empty
        if pd.isna(reference) or pd.isna(hypothesis):
            skipped_rows.append(row_id)
            continue
        
        reference = str(reference).strip()
        hypothesis = str(hypothesis).strip()
        
        # Skip if either is empty after stripping
        if not reference or not hypothesis:
            skipped_rows.append(row_id)
            continue
        
        # Calculate metrics
        cer = calculate_cer(reference, hypothesis)
        wer = calculate_wer(reference, hypothesis)
        bleu = calculate_bleu(reference, hypothesis)
        meteor = calculate_meteor(reference, hypothesis)
        rouge = calculate_rouge(reference, hypothesis)
        
        results.append({
            'ID': row_id,
            'BLEU': round(bleu, 2),
            'ROUGE-1': round(rouge['ROUGE-1'], 2),
            'ROUGE-2': round(rouge['ROUGE-2'], 2),
            'ROUGE-L': round(rouge['ROUGE-L'], 2),
            'METEOR': round(meteor, 2),
            'CER': round(cer, 2),
            'WER': round(wer, 2)
        })
    
    # Print summary of skipped rows
    if skipped_rows:
        print(f"\n⚠ Skipped {len(skipped_rows)} rows due to NaN or empty values:")
        print(f"  Row IDs: {skipped_rows[:10]}{'...' if len(skipped_rows) > 10 else ''}")
    
    print(f"\n✓ Successfully evaluated {len(results)} translation pairs out of {rows_to_process} rows with matching IDs.")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    metrics = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'CER', 'WER']
    
    stats_data = []
    for metric in metrics:
        stats_data.append({
            'Metric': metric,
            'Mean': round(results_df[metric].mean(), 2),
            'Std': round(results_df[metric].std(), 2),
            'Min': round(results_df[metric].min(), 2),
            'Max': round(results_df[metric].max(), 2),
            'Median': round(results_df[metric].median(), 2)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Get output prefix from gemini CSV filename
    output_folder, output_prefix = get_output_prefix(gemini_csv_path)
    
    return results_df, stats_df, output_folder, output_prefix

# Example usage:
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    human_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\evaluation\\translation-csv\\human\\Sylheti_human.csv"
    #gemini_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\Translation_Output\\GPT\\Translated_Chatgaiyan_10_shot\\chatgaiyan_10_shot_gpt.csv"
    gemini_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\Translation_Output\\GPT\\Translated_Sylheti_10_shot\\Sylheti_10_shot_gpt.csv"
    #gemini_csv = "D:\\coding\\projects\\Bangla-to-Regional-Languages-Dataset\\Translation_Output\\Qwen\\chatgaiyan_0_shot_qwen.csv"
    
    # Process translations
    # Parameters:
    #   - text_column_index: Column index for translation text (default: 1)
    #   - id_column_index: Column index for ID (default: 0)
    #   - max_rows: Maximum rows to evaluate (default: None for all rows)
    
    # Example 1: Evaluate all matching rows
    detailed_results, statistics, output_folder, output_prefix = process_translations(
        human_csv, 
        gemini_csv, 
        text_column_index=1,
        id_column_index=0,
        max_rows=600  # Use None to evaluate all matching rows
    )
    
    # Example 2: Evaluate only first 100 matching rows
    # detailed_results, statistics, output_folder, output_prefix = process_translations(
    #     human_csv, 
    #     gemini_csv, 
    #     text_column_index=1,
    #     id_column_index=0,
    #     max_rows=100  # Limit to first 100 rows
    # )

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
    print("SIMILARITY SCORES (Higher is better, 100 = perfect match):")
    print("  • BLEU: Measures n-gram overlap (0-100)")
    print("  • ROUGE-1: Unigram overlap (0-100)")
    print("  • ROUGE-2: Bigram overlap (0-100)")
    print("  • ROUGE-L: Longest common subsequence (0-100)")
    print("  • METEOR: Considers synonyms and word order (0-100)")
    print("\nERROR RATES (Lower is better, 0 = perfect match):")
    print("  • CER: Character Error Rate (0-100+)")
    print("  • WER: Word Error Rate (0-100+)")
    print("\nNOTE: Low scores indicate significant differences between translations.")
    print("This is normal if the translations use different words or phrasing.")
    print("=" * 100)