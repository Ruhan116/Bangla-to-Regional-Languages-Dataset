import pandas as pd
import sacrebleu
import os
from tabulate import tabulate

def calculate_sacrebleu(csv_ref, csv_hyp, limit=500, hyp_text_column=1, hyp_has_header=False, ref_text_column=1):
    # Load CSVs - reference and hypothesis may have different structures
    print(f"Loading reference CSV: {csv_ref}")
    df_ref_raw = pd.read_csv(csv_ref, header=None, quoting=1)  # Add quoting for proper parsing
    print(f"Reference CSV shape: {df_ref_raw.shape}")
    print(f"Reference CSV first few rows:\n{df_ref_raw.head()}")
    
    # Select appropriate columns from reference CSV
    df_ref = df_ref_raw[[0, ref_text_column]].copy()
    df_ref.columns = ["id", "text"]
    
    # Clean data: Remove rows with missing text values
    print(f"Reference CSV before cleaning: {df_ref.shape}")
    df_ref = df_ref.dropna()  # Remove rows where any column is NaN
    df_ref['text'] = df_ref['text'].astype(str)  # Ensure text is string type
    # Remove any remaining empty strings or strings that represent NaN
    df_ref = df_ref[df_ref['text'].str.strip() != '']
    df_ref = df_ref[df_ref['text'] != 'nan']
    print(f"Reference CSV after cleaning: {df_ref.shape}")
    print(f"Reference CSV after column selection first few rows:\n{df_ref.head()}")
    
    print(f"\nLoading hypothesis CSV: {csv_hyp}")
    if hyp_has_header:
        df_hyp = pd.read_csv(csv_hyp, header=0)
        print(f"Hypothesis CSV shape: {df_hyp.shape}")
        print(f"Hypothesis CSV first few rows:\n{df_hyp.head()}")
        # Get column names for reference
        print(f"Hypothesis CSV columns: {list(df_hyp.columns)}")
        # Extract ID (first column) and text (specified column index)
        df_hyp = df_hyp.iloc[:, [0, hyp_text_column]].copy()
    else:
        df_hyp = pd.read_csv(csv_hyp, header=None)
        print(f"Hypothesis CSV shape: {df_hyp.shape}")
        print(f"Hypothesis CSV first few rows:\n{df_hyp.head()}")
        # Extract ID (column 0) and text (specified column) from hypothesis
        df_hyp = df_hyp[[0, hyp_text_column]].copy()
    
    df_hyp.columns = ["id", "text"]
    
    # Clean data: Remove rows with missing text values
    print(f"Hypothesis CSV before cleaning: {df_hyp.shape}")
    df_hyp = df_hyp.dropna()  # Remove rows where any column is NaN
    df_hyp['text'] = df_hyp['text'].astype(str)  # Ensure text is string type
    # Remove any remaining empty strings or strings that represent NaN
    df_hyp = df_hyp[df_hyp['text'].str.strip() != '']
    df_hyp = df_hyp[df_hyp['text'] != 'nan']
    print(f"Hypothesis CSV after cleaning: {df_hyp.shape}")
    print(f"Hypothesis CSV after column selection first few rows:\n{df_hyp.head()}")

    # Convert IDs to string for matching
    df_ref["id"] = df_ref["id"].astype(str)
    df_hyp["id"] = df_hyp["id"].astype(str)
    
    print(f"\nUnique IDs in reference: {len(df_ref['id'].unique())}")
    print(f"Unique IDs in hypothesis: {len(df_hyp['id'].unique())}")
    print(f"Sample reference IDs: {list(df_ref['id'].head())}")
    print(f"Sample hypothesis IDs: {list(df_hyp['id'].head())}")
    
    # Merge on ID (inner join ensures only matched ids are kept)
    df = pd.merge(df_ref, df_hyp, on="id", suffixes=("_ref", "_hyp"))
    print(f"\nAfter merge shape: {df.shape}")

    # Take only first `limit` rows
    df = df.head(limit)
    
    # Show first few merged rows for inspection
    print(f"\nFirst 5 merged rows:")
    print(df[['id', 'text_ref', 'text_hyp']].head())

    # Convert to lists for sacreBLEU
    references = [[r] for r in df["text_ref"].tolist()]  # list of list (corpus BLEU format)
    hypotheses = df["text_hyp"].tolist()

    if len(hypotheses) == 0 or len(references) == 0:
        raise ValueError("No data to evaluate: hypotheses or references are empty.")

    # Compute sacreBLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, list(zip(*references)))

    print("Merged DataFrame shape:", df.shape)
    print(f"SacreBLEU Score: {bleu.score:.2f}")
    
    return bleu.score

def evaluate_multiple_hypotheses(ref_csv, hyp_csv_list, hyp_text_column_list, hyp_has_header_list, limit=500, ref_text_column=1):
    """
    Evaluate multiple hypothesis CSVs against a single reference CSV.
    
    Args:
        ref_csv: Path to reference CSV
        hyp_csv_list: List of paths to hypothesis CSVs
        hyp_text_column_list: List of column indices for hypothesis text
        hyp_has_header_list: List of boolean values indicating if CSV has headers
        limit: Maximum number of rows to evaluate
        ref_text_column: Column index for reference text
    
    Returns:
        List of dictionaries containing results for each hypothesis
    """
    results = []
    
    for i, (hyp_csv, hyp_text_column, hyp_has_header) in enumerate(zip(hyp_csv_list, hyp_text_column_list, hyp_has_header_list)):
        print(f"\n{'='*80}")
        print(f"EVALUATING HYPOTHESIS {i+1}/{len(hyp_csv_list)}: {os.path.basename(hyp_csv)}")
        print(f"{'='*80}")
        
        try:
            score = calculate_sacrebleu(
                csv_ref=ref_csv,
                csv_hyp=hyp_csv,
                limit=limit,
                hyp_text_column=hyp_text_column,
                hyp_has_header=hyp_has_header,
                ref_text_column=ref_text_column
            )
            
            result = {
                'file': os.path.basename(hyp_csv),
                'score': score,
                'status': 'Success'
            }
            
        except Exception as e:
            print(f"ERROR: Failed to evaluate {hyp_csv}")
            print(f"Error details: {str(e)}")
            result = {
                'file': os.path.basename(hyp_csv),
                'score': 0.0,
                'status': f'Error: {str(e)[:50]}...'
            }
        
        results.append(result)
        print(f"\nResult for {os.path.basename(hyp_csv)}: {result['score']:.2f}")
    
    return results

def print_results_table(results):
    """
    Print results in a formatted table.
    """
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Prepare data for tabulation
    table_data = []
    for result in results:
        table_data.append([
            result['file'],
            f"{result['score']:.2f}" if result['status'] == 'Success' else 'ERROR',
            result['status']
        ])
    
    # Print table
    headers = ['File Name', 'BLEU Score', 'Status']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Print summary statistics
    successful_results = [r for r in results if r['status'] == 'Success']
    if successful_results:
        scores = [r['score'] for r in successful_results]
        print(f"\nSummary Statistics:")
        print(f"- Files evaluated: {len(results)}")
        print(f"- Successful evaluations: {len(successful_results)}")
        print(f"- Average BLEU score: {sum(scores)/len(scores):.2f}")
        print(f"- Best score: {max(scores):.2f} ({[r['file'] for r in successful_results if r['score'] == max(scores)][0]})")
        print(f"- Worst score: {min(scores):.2f} ({[r['file'] for r in successful_results if r['score'] == min(scores)][0]})")


if __name__ == "__main__":
    # Reference CSV file
    ref_csv = r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\Test_Samples_Sylheti_Human.csv"
    
    # List of hypothesis CSV files to evaluate (from filtered_hyps folder)
    hyp_csv_list = [
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_0_shot_gemini_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_0_shot_gemma_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_0_shot_gpt_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_0_shot_qwen_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_gemini_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_gemma_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_gpt_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_qwen_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_CoT_gemma_cleaned_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_5_shot_CoT_qwen_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_10_shot_gemini_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_10_shot_gemma_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_10_shot_gpt_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\Sylheti_10_shot_qwen_filtered.csv",
        r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps\claude_0_shot_train_filtered.csv"
    ]
    
    # Configuration for each hypothesis file - all have text in column 1 and no headers
    hyp_text_column_list = [1] * len(hyp_csv_list)  # Column index for hypothesis text (all files have text in column 1)
    hyp_has_header_list = [False] * len(hyp_csv_list)  # All files have no headers
    
    # Evaluate multiple hypothesis files
    results = evaluate_multiple_hypotheses(
        ref_csv=ref_csv,
        hyp_csv_list=hyp_csv_list,
        hyp_text_column_list=hyp_text_column_list,
        hyp_has_header_list=hyp_has_header_list,
        limit=500,
        ref_text_column=1
    )
    
    # Print results table
    print_results_table(results)
