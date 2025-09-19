import os
import csv

def split_csv(input_path, output_dir, prefix, lines_per_file=200):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        file_count = 0
        rows = []
        for i, row in enumerate(reader, 1):
            rows.append(row)
            if i % lines_per_file == 0:
                out_path = os.path.join(output_dir, f"{prefix}{file_count}.csv")
                with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(rows)
                rows = []
                file_count += 1
        # Write remaining rows
        if rows:
            out_path = os.path.join(output_dir, f"{prefix}{file_count}.csv")
            with open(out_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                writer.writerows(rows)

if __name__ == "__main__":
    split_csv('BangaCHQ/train.csv', 'Split_Dataset', 'train')
    split_csv('BangaCHQ/test.csv', 'Split_Dataset', 'test')
    split_csv('BangaCHQ/valid.csv', 'Split_Dataset', 'valid')
