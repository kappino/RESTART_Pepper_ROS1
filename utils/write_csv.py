import csv
import os

def write_csv(name_file, mode, field_names, row):
    file_exists = os.path.isfile(name_file)
    with open(name_file, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)