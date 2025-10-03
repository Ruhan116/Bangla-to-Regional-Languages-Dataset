import pandas as pd
import re

# Read the CSV file
df = pd.read_csv('Bangla CHQ Prantik/Bangla CHQ Prantik.csv')

def calculate_stats(column_name):
    """Calculate statistics for a given column"""
    # Filter out null/empty values
    text_entries = df[column_name].dropna()
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
    # Split by common Bangla sentence delimiters (। and ?)
    all_sentences = set()
    for text in text_entries:
        sentences = re.split(r'[।?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        all_sentences.update(sentences)
    unique_sentence_count = len(all_sentences)
    
    return {
        'mean_char_length': mean_char_length,
        'max_char_length': max_char_length,
        'min_char_length': min_char_length,
        'mean_word_count': mean_word_count,
        'max_word_count': max_word_count,
        'min_word_count': min_word_count,
        'unique_word_count': unique_word_count,
        'unique_sentence_count': unique_sentence_count
    }

# Calculate statistics for both columns
bangla_stats = calculate_stats('Bangla CHQ')
sylheti_stats = calculate_stats('Sylheti CHQ')

# Display results
print("=" * 50)
print("BANGLA CHQ STATISTICS")
print("=" * 50)
print(f"Mean Character Length: {bangla_stats['mean_char_length']:.2f}")
print(f"Max Character Length: {bangla_stats['max_char_length']}")
print(f"Min Character Length: {bangla_stats['min_char_length']}")
print(f"\nMean Word Count: {bangla_stats['mean_word_count']:.2f}")
print(f"Max Word Count: {bangla_stats['max_word_count']}")
print(f"Min Word Count: {bangla_stats['min_word_count']}")
print(f"\nUnique Word Count: {bangla_stats['unique_word_count']}")
print(f"Unique Sentence Count: {bangla_stats['unique_sentence_count']}")

print("\n" + "=" * 50)
print("SYLHETI CHQ STATISTICS")
print("=" * 50)
print(f"Mean Character Length: {sylheti_stats['mean_char_length']:.2f}")
print(f"Max Character Length: {sylheti_stats['max_char_length']}")
print(f"Min Character Length: {sylheti_stats['min_char_length']}")
print(f"\nMean Word Count: {sylheti_stats['mean_word_count']:.2f}")
print(f"Max Word Count: {sylheti_stats['max_word_count']}")
print(f"Min Word Count: {sylheti_stats['min_word_count']}")
print(f"\nUnique Word Count: {sylheti_stats['unique_word_count']}")
print(f"Unique Sentence Count: {sylheti_stats['unique_sentence_count']}")