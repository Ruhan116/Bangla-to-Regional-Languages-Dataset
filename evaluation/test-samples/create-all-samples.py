import pandas as pd
import os

# Paths
ref_path = r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\Test_Samples_Sylheti_Human.csv"
hyp_paths = [
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Claude\Translated_Sylheti_0_Shot\train.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemini\Translated_Sylhet_0_Shot\Sylheti_0_shot_gemini.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemini\Translated_Sylhet_5_Shot\Sylheti_5_shot_gemini.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemini\Translated_Sylhet_10_Shot\Sylheti_10_shot_gemini.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemma\Sylheti_0_shot_gemma.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemma\Sylheti_5_shot_CoT_gemma_cleaned.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemma\Sylheti_5_shot_gemma.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gemma\Sylheti_10_shot_gemma.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gpt\Translated_Sylheti_0_Shot\Sylheti_0_shot_gpt.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gpt\Translated_Sylheti_5_Shot\Sylheti_5_shot_gpt.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Gpt\Translated_Sylheti_10_Shot\Sylheti_10_shot_gpt.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Qwen\Sylheti_0_shot_qwen.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Qwen\Sylheti_5_shot_CoT_qwen.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Qwen\Sylheti_5_shot_qwen.csv",
    r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\Translation_Output\Qwen\Sylheti_10_shot_qwen.csv"
]
output_dir = r"D:\Thesis\UwU\BLP\Bangla-to-Regional-Languages-Dataset\evaluation\test-samples\filtered_hyps"
os.makedirs(output_dir, exist_ok=True)

# Read reference sentences
ref_df = pd.read_csv(ref_path, dtype=str, keep_default_na=False)
ref_sentences = ref_df.iloc[:, 0].astype(str).str.strip().str.replace('\r\n', '\n').str.replace('\r', '\n').tolist()

for hyp_path in hyp_paths:
    try:
        # Check if file exists
        if not os.path.exists(hyp_path):
            print(f"Warning: File not found: {hyp_path}")
            continue
            
        # Try reading CSV with different parameters if first attempt fails
        try:
            hyp_df = pd.read_csv(hyp_path, dtype=str, keep_default_na=False)
        except pd.errors.ParserError:
            # Try with different parsing options for malformed CSV
            print(f"Warning: Parser error in {os.path.basename(hyp_path)}, trying alternative parsing...")
            try:
                hyp_df = pd.read_csv(hyp_path, dtype=str, keep_default_na=False, on_bad_lines='skip')
                print(f"Warning: Some malformed lines were skipped in {os.path.basename(hyp_path)}")
            except Exception:
                # Try reading line by line as a last resort
                print(f"Warning: Using manual parsing for {os.path.basename(hyp_path)}")
                lines = []
                with open(hyp_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:  # Keep lines with at least 2 columns
                            lines.append(parts[:2])  # Take only first 2 columns
                        else:
                            print(f"Skipping malformed line {line_num} in {os.path.basename(hyp_path)}")
                hyp_df = pd.DataFrame(lines, columns=['col1', 'col2'])
        
        print(f"Processing: {os.path.basename(hyp_path)} ({hyp_df.shape[0]} rows, {hyp_df.shape[1]} columns)")
        
        # Normalize all columns for matching (using map instead of deprecated applymap)
        hyp_df_norm = hyp_df.map(lambda x: str(x).strip().replace('\r\n', '\n').replace('\r', '\n'))

        # Create a mapping for efficient lookup
        hyp_sentence_to_idx = {}
        for idx, row in hyp_df_norm.iterrows():
            sentence = row.iloc[0]
            if sentence not in hyp_sentence_to_idx:
                hyp_sentence_to_idx[sentence] = idx

        # Find matches efficiently
        matches = []
        not_found_count = 0
        for idx, ref_sentence in enumerate(ref_sentences):
            if ref_sentence in hyp_sentence_to_idx:
                matches.append(hyp_sentence_to_idx[ref_sentence])
            else:
                not_found_count += 1
        
        if not_found_count > 0:
            print(f"Warning: {not_found_count} reference sentences not found in {os.path.basename(hyp_path)}")
        
        if not matches:
            print(f"Error: No matches found for {os.path.basename(hyp_path)}")
            continue

        # Filter hyp_df by matched indices
        filtered_hyp_df = hyp_df.iloc[matches, :].copy()

        # Select columns: if 2 columns, both; if more, first and last
        if filtered_hyp_df.shape[1] == 2:
            filtered_hyp_df = filtered_hyp_df.iloc[:, [0, 1]]
        elif filtered_hyp_df.shape[1] > 2:
            filtered_hyp_df = filtered_hyp_df.iloc[:, [0, -1]]

        # Save filtered hyp
        out_name = os.path.splitext(os.path.basename(hyp_path))[0] + "_filtered.csv"
        out_path = os.path.join(output_dir, out_name)
        filtered_hyp_df.to_csv(out_path, index=False, encoding='utf-8-sig')

        print(f"Filtered hyp saved to: {out_path} ({filtered_hyp_df.shape[0]} rows matched)")
        
    except Exception as e:
        print(f"Error processing {hyp_path}: {str(e)}")
        continue