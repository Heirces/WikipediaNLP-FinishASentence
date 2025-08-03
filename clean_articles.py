import json
import re
import zipfile
import os
import spacy
pattern_table = r"\{\|.*?\|\}"
pattern_cleanup_table_leftovers = r"\|.*?\|\}"
pattern_table_cell_cleanup = r"\|.*?\|"
pattern_carrot_text = r"\<.*?\>"
pattern_titles = r"={2,6}[^=].*?[^=]={2,6}"
pattern_Categories_at_end_of_articles = r"Category:.*"
pattern_URLs = r"(https?://)?([a-zA-Z0-9\-]+\.)+[a-z]{2,6}(/[^\s]*)?"
pattern_quotes = r'["“”‘’\'`]'
def create_clean_text_files_from_archive_zip():
    with zipfile.ZipFile("archive.zip", 'r') as wikipedia_archive:
        clean_text_file_name_base = "cleaned_text_files/clean_text"     # Every 30 cleaned json files will be appended to a single clean_text file
        clean_text_file_number = 1 
        number_of_json_files = 0
        for file in wikipedia_archive.namelist():
            if file.endswith('.json'):
                number_of_json_files += 1
                if number_of_json_files % 30 == 0:
                    clean_text_file_number += 1
                if number_of_json_files % 10 == 0:
                    print(f"Processed {number_of_json_files} JSON files...")
                clean_text_file_name = clean_text_file_name_base + str(clean_text_file_number) + ".txt"
                with wikipedia_archive.open(file) as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Skipping corrupted json file: {file}")
                        continue
                    for article in data:
                        article_text = article["text"]
                        clean_text = re.sub(pattern_table, '', article_text)
                        clean_text = re.sub(pattern_cleanup_table_leftovers, '', clean_text)
                        clean_text = re.sub(pattern_table_cell_cleanup, '', clean_text)
                        clean_text = re.sub(pattern_carrot_text, '', clean_text)
                        clean_text = re.sub(pattern_titles, '', clean_text)
                        clean_text = re.sub(pattern_Categories_at_end_of_articles, "", clean_text, flags=re.DOTALL)
                        clean_text = re.sub(r"\*", '', clean_text)
                        clean_text = re.sub(pattern_URLs, '', clean_text)
                        clean_text = re.sub(pattern_quotes, '', clean_text)
                        
                        with open(clean_text_file_name, "a", encoding="utf-8") as out_file:
                            out_file.write(clean_text.strip() + "\n\n")
    
def reclean_the_clean_text_files(clean_text_file):
    with open(clean_text_file, "r", encoding="utf-8") as working_file:
        text = working_file.read()
    cleaned_text = re.sub(pattern_quotes, '', text)
    print("Halfway Done")
    with open(clean_text_file, "w", encoding="utf-8") as final_file:
        final_file.write(cleaned_text)

def reclean_all_the_text_files():
    directory = r"C:\Users\Garre\OneDrive\Desktop\WGU\Capstone\cleaned_text_files"
    for filename in os.listdir(directory):
        if filename != "clean_text1.txt":
            filepath = os.path.join(directory, filename)
            print(f"Cleaning: {filename}")
            try:
                reclean_the_clean_text_files(filepath)
            except Exception as e:
                print(f=
                "Error processing {filename}: {e}")


