import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('Sylheti_human  merged_clean.csv')

# Get the text column (second column)
text_column = df.columns[1]

# Filter out null/empty values
text_entries = df[text_column].dropna()
text_entries = text_entries[text_entries.str.strip() != '']

# Calculate character length statistics
char_lengths = text_entries.str.len()
mean_char_length = char_lengths.mean()
max_char_length = char_lengths.max()
min_char_length = char_lengths.min()

# Calculate word count statistics
word_counts = text_entries.apply(lambda x: len(x.strip().split()))
mean_word_count = word_counts.mean()
max_word_count = word_counts.max()
min_word_count = word_counts.min()

# Calculate unique word count (across all entries)
all_words = set()
for text in text_entries:
    words = text.strip().split()
    all_words.update(words)
unique_word_count = len(all_words)

# Calculate unique sentence count
# Split by common sentence delimiters (। and ?)
all_sentences = set()
for text in text_entries:
    sentences = re.split(r'[।?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    all_sentences.update(sentences)
unique_sentence_count = len(all_sentences)

# Display results
print("========== TEXT STATISTICS ==========\n")
print("Character Length Statistics:")
print(f"  Mean Character Length: {mean_char_length:.2f}")
print(f"  Max Character Length: {max_char_length}")
print(f"  Min Character Length: {min_char_length}")

print("\nWord Count Statistics:")
print(f"  Mean Word Count: {mean_word_count:.2f}")
print(f"  Max Word Count: {max_word_count}")
print(f"  Min Word Count: {min_word_count}")

print("\nUnique Counts:")
print(f"  Unique Word Count: {unique_word_count}")
print(f"  Unique Sentence Count: {unique_sentence_count}")
print("\n====================================")