import csv

input_file = "gemma_5_shot_sylheti_CoT.csv"
output_file = "gemma_5_shot_sylheti_CoT_cleaned.csv"

# Labels to strip from the beginning of the final line (case-insensitive)
LABEL_PREFIXES = ("chittagonian:", "sylheti:", "indices", "sylheti")

def strip_known_label(s: str) -> str:
    s_stripped = s.lstrip()
    lower = s_stripped.lower()
    for p in LABEL_PREFIXES:
        if lower.startswith(p.lower()):
            return s_stripped[len(p):].strip()
    return s.strip()

with open(input_file, "r", newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Skip header row and write new header
    next(reader, None)
    writer.writerow(["record_id", "cleaned_text"])

    for row in reader:
        if not row:
            continue

        # Handle both 1-col and 2+-col CSVs
        if len(row) == 1:
            record_id = ""
            text = row[0]
        else:
            record_id = row[0].strip()
            text = row[1] if len(row) > 1 else ""

        if not text:
            continue

        # Split into non-empty, trimmed lines (handles embedded newlines inside quoted CSV field)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            continue

        # Take the last non-empty line and remove known label prefixes if present
        final_line = strip_known_label(lines[-1])
        
        # Clean up any remaining artifacts
        final_line = final_line.strip('"').strip()

        # Write ID and cleaned text
        writer.writerow([record_id, final_line])
