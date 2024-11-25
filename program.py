import re
from pydantic import BaseModel, EmailStr
import requests
import random
import numpy as np
import pandas as pd
from tkinter import *
from time import sleep
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
import matplotlib.pyplot as plt
from tkcalendar import DateEntry
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from easygui.boxes.fileopen_box import tk
from tkinter import filedialog, messagebox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os

subject = ""
start_date = ""
end_date = ""
resolved_complaints_count = 0
received_complaints_count = 0


# Define df as a global variable
df = pd.DataFrame(columns=["Tweet", "User Name", "Tweet Date"])

def submit_input():
    global subject, start_date, end_date
    subject = id_value.get()
    start_date = entry_start_date.get()
    end_date = entry_end_date.get()

    # # Add the @ or # prefix based on the selected subject type
    # if subject_type.get() == "username" and not subject.startswith('@'):
    #     subject = '@' + subject
    # elif subject_type.get() == "hashtag" and not subject.startswith('#'):
    #     subject = '#' + subject

    print(f"Subject: {subject}, Start Date: {start_date}, End Date: {end_date}, Folder Path: {folder_path}")

    # Close the GUI window after submitting the input
    root.destroy()


def scrape_tweets():
    global df, received_complaints_count, resolved_complaints_count
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    driver = webdriver.Chrome()
    driver.maximize_window()

    driver.get("https://twitter.com/login")

    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//input[@name='text']")))
    username = driver.find_element(By.XPATH, "//input[@name='text']")
    username.send_keys("MdSajeer4690")

    next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
    next_button.click()

    password = WebDriverWait(driver, 60).until(
        EC.visibility_of_element_located((By.XPATH, "//input[@name='password']")))
    password.send_keys('Sajeer@123')

    log_in = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
    log_in.click()
    sleep(5)  # Increased wait time after login

    if subject_type.get() == "username":
        # Code for username
        driver.get(f"https://twitter.com/{subject}/with_replies")
    elif subject_type.get() == "hashtag":
        # Code for hashtag
        driver.get(f"https://twitter.com/search?q=%23{subject}&src=typed_query&f=live")
    else:
        print("Invalid subject type. Please select either username or hashtag.")
        driver.quit()
        return None, None

    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//article[@data-testid="tweet"]')))
    sleep(4)

    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_distance = 1200  # Increased initial scroll distance
    # small_scroll = 100
    # large_scroll = 1000
    # no_new_tweets_count = 0
    # max_no_new_tweets = 5
    resolved_complaints_count = 0
    received_complaints_count = 0
    tweet_data = []
    reply_data = []
    seen_tweets = set()
    strikes = 0
    max_strikes = 5  # Increased max strikes

    while True:
        # last_tweet_count = len(tweet_data)
        driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
        sleep(3)  # Increased wait time after scrolling

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        tweet_elements = soup.find_all("article", {"data-testid": "tweet"})

        previous_received_count = received_complaints_count
        previous_resolved_count = resolved_complaints_count
        previous_tweet_count = len(tweet_data)

        oldest_tweet_date = None

        for tweet in tweet_elements:
            tweet_date_tag = tweet.find("time", {"datetime": True})
            tweet_date_text = tweet_date_tag["datetime"] if tweet_date_tag else None

            if tweet_date_text:
                tweet_date = datetime.strptime(tweet_date_text, "%Y-%m-%dT%H:%M:%S.%fZ")

                if oldest_tweet_date is None or tweet_date < oldest_tweet_date:
                    oldest_tweet_date = tweet_date

                if start_datetime <= tweet_date <= end_datetime + timedelta(days=1):
                    user_name_tag = tweet.find("div", {"data-testid": "User-Name"})
                    user_name = user_name_tag.text.strip() if user_name_tag else "User name not found"

                    tweet_text_tag = tweet.find("div", {"data-testid": "tweetText"})
                    tweet_text = tweet_text_tag.text.strip() if tweet_text_tag else "Tweet text not found"

                    image_links = []
                    image_tags = tweet.find_all("div", {"data-testid": "tweetPhoto"})
                    for image_tag in image_tags:
                        img_tag = image_tag.find("img")
                        if img_tag and "src" in img_tag.attrs:
                            image_link = img_tag["src"]
                            image_links.append(image_link)

                    if (user_name, tweet_text, tweet_date_text) in seen_tweets:
                        continue

                    if "Thank you for your tweet. It has been sent to the concerned officer for follow-up action." in tweet_text or \
                            "எங்களுக்கு தெரியப்படுத்தியமைக்கு நன்றி. தாங்கள் தெரிவித்த தகவல் தொடர் நடவடிக்கைக்காக சம்பந்தப்பட்ட அதிகாரிக்கு அனுப்பப்பட்டுள்ளது." in tweet_text or \
                            "Thank you for informing us. It has been sent to the concerned officer for follow-up action." in tweet_text:
                        received_complaints_count += 1
                        reply_data.append({
                            'Reply': tweet_text,
                            'Reply User Name': user_name,
                            'Reply Date': tweet_date_text,
                        })
                    elif "Your tweet was verified and action taken." in tweet_text or \
                            "Action taken on your tweet. Thank you for informing us." in tweet_text:
                        resolved_complaints_count += 1
                        reply_data.append({
                            'Reply': tweet_text,
                            'Reply User Name': user_name,
                            'Reply Date': tweet_date_text,
                        })
                    else:
                        tweet_data.append({
                            'Tweet': tweet_text,
                            'User Name': user_name,
                            'Tweet Date': tweet_date_text,
                            'Image Links': image_links,
                        })

                    seen_tweets.add((user_name, tweet_text, tweet_date_text))

        if (received_complaints_count == previous_received_count and
                resolved_complaints_count == previous_resolved_count and
                len(tweet_data) == previous_tweet_count):
            strikes += 1
        else:
            strikes = 0

        print(f"Strikes: {strikes}/{max_strikes}")
        print(f"Tweets collected: {len(tweet_data)}")
        print(f"Oldest tweet date: {oldest_tweet_date}")

        if strikes >= max_strikes:
            print("Stopping condition met.")
            break

        # new_height = driver.execute_script("return document.body.scrollHeight")
        # if new_height == last_height:
        #     driver.execute_script(f"window.scrollBy(0, {large_scroll});")
        #     sleep(3)
        #     new_height = driver.execute_script("return document.body.scrollHeight")
        #     if new_height == last_height:
        #         print("Reached end of page.")
        #         break

        # last_height = new_height
        # scroll_distance += 500  # Increased scroll increment

    driver.quit()

    print(f"Number of tweets collected: {len(tweet_data)}")
    print(f"Number of replies collected: {len(reply_data)}")

    df = pd.DataFrame(tweet_data)
    df_replies = pd.DataFrame(reply_data)

    file_name = r"Replies.xlsx"
    excel_file = os.path.join(folder_path, file_name)
    df_replies.to_excel(f"{excel_file}", index=False)

    print(df)

    return df, df_replies


def browse_folders():
    # Use askdirectory for selecting folders
    global folder_path  # Declare folder_path as global before using it
    folder_path = filedialog.askdirectory(title="Select a folder")
    if folder_path:
        entry4.delete(0, END)
        entry4.insert(0, folder_path)

# root = Tk()
# root.title("SentiNal")
# root.geometry("500x200")
#
# # Create separate variables for input data
# id_value = StringVar()


# Initialize folder path as an empty string
folder_path = ""
def set_focus(event):
    # Automatically place cursor in the first entry field
    entry1.focus_set()

root = tk.Tk()
root.title("SentiNel Dashboard")
root.geometry("500x250")

# Define styles
label_style = {"font": ("Helvetica", 10), "bg": "#F0F0F0"}
entry_style = {"font": ("Helvetica", 10)}
button_style = {"font": ("Helvetica", 10)}

# Entry variables
id_value = tk.StringVar()
destination_path = tk.StringVar()

# Label and Entry widgets
label1 = tk.Label(root, text="Enter the ID:", **label_style)
label1.grid(row=0, column=0, padx=5, pady=5)

entry1 = tk.Entry(root, textvariable=id_value, **entry_style)
entry1.grid(row=0, column=1, padx=5, pady=5)
entry1.bind("<Button-1>", set_focus)  # Automatically set focus on entry1 when clicked

subject_type = tk.StringVar(value="username")
tk.Radiobutton(root, text="Username", variable=subject_type, value="username").grid(row=5, column=0, padx=5, pady=5)
tk.Radiobutton(root, text="Hashtag", variable=subject_type, value="hashtag").grid(row=5, column=1, padx=5, pady=5)

label_start_date = tk.Label(root, text="Select start date:", **label_style)
label_start_date.grid(row=1, column=0, padx=5, pady=5)
entry_start_date = DateEntry(root, date_pattern='yyyy-mm-dd', **entry_style)
entry_start_date.grid(row=1, column=1, padx=5, pady=5)

label_end_date = tk.Label(root, text="Select end date:", **label_style)
label_end_date.grid(row=2, column=0, padx=5, pady=5)
entry_end_date = DateEntry(root, date_pattern='yyyy-mm-dd', **entry_style)
entry_end_date.grid(row=2, column=1, padx=5, pady=5)

label4 = tk.Label(root, text="Destination Path:", **label_style)
label4.grid(row=3, column=0, padx=5, pady=5)

entry4 = tk.Entry(root, textvariable=destination_path, **entry_style)
entry4.grid(row=3, column=1, padx=5, pady=5)

button1 = tk.Button(root, text="Browse folder...", command=browse_folders, **button_style)
button1.grid(row=3, column=2, padx=5, pady=5)

entry1.focus_set()


def submit_data(id, start_date, end_date, folder_path):
    # Process the collected data here
    print(f"ID: {id}, Start Date: {start_date}, End Date: {end_date}, Folder Path: {folder_path}")
    # You can perform actions like saving data based on the selected folder here

button2 = Button(root, text="Submit", command=submit_input)
button2.grid(row=4, columnspan=2, padx=5, pady=5)

root.mainloop()


scrape_tweets()

def translate_tweets(df, column_to_translate="Tweet", target_lang="en_XX"):
    # Load the model and tokenizer

    # Load model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # Set source language for the tokenizer
    tokenizer.src_lang = "ta_IN"

    def translate_text(text, target_lang):
        encoded_text = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    # Translate each text in the selected column and create a new column for translated text
    df["translated_text"] = df[column_to_translate].apply(lambda x: translate_text(x, target_lang))

    return df
translate_tweets(df)
def perform_sentiment_analysis(df):

    # Function to categorize sentiment
    def categorize_sentiment(sentiment, text):
        lower_text = text.lower()  # Convert text to lower case for case-insensitive matching
        if any(word in lower_text for word in
               ["please take action", "still no action", "no parking zone", "making traffic jams",
                "defective number plate", "wrong side driving"]):
            return "complaint"
        elif any(char.isdigit() for char in text):
            return "complaint"
        elif sentiment == "positive":
            return "positive"
        elif sentiment == "negative":
            return "negative"
        else:
            return "unknown"

    # Preprocess text (username and link placeholders)
    def preprocess(text):
        if isinstance(text, str):  # Check if text is a string
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            return " ".join(new_text)
        else:
            return ""  # Return empty string for NaN values

    # -----------Sentiment_Process-----------------#

    task = 'sentiment'
    MODEL = f"twitter-roberta-base-sentiment"

    # Load the tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Download label mapping
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    response = requests.get(mapping_link)
    lines = response.text.split("\n")
    print("Labels from URL:", lines)
    labels = [line.split('\t')[1] for line in lines if len(line.split('\t')) > 1]
    print(labels)

    # Handle NaN values in the 'Tweet' column
    df['translated_text'] = df['translated_text'].fillna("")  # Replace NaN values with empty strings

    # Perform sentiment analysis for each text in the specified column
    sentiments = []
    for text in df['translated_text']:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores, axis=-1)[::-1]  # Ensure proper sorting
        print("Ranking:", ranking)  # Print the ranking obtained

        # Check if ranking is a scalar index
        if np.isscalar(ranking):
            top_sentiment_index = ranking
        else:
            # Flatten the ranking array to handle nested structures
            flat_ranking = ranking.flatten()
            top_sentiment_index = flat_ranking[0]

        # Map the top sentiment index to the corresponding label
        if 0 <= top_sentiment_index < len(labels):
            top_sentiment_label = labels[top_sentiment_index]
        else:
            top_sentiment_label = "unknown"

        sentiments.append(top_sentiment_label)

    # -----------Sentiment_Process_ends-----------------#

    # Add the sentiments to the DataFrame
    df['sentiment'] = sentiments

    # Categorize sentiment based on presence of numeric characters
    df['sentiment_category'] = df.apply(lambda row: categorize_sentiment(row['sentiment'], row['translated_text']), axis=1)

    return df
perform_sentiment_analysis(df)
# def most_tweets_by_sentiment(sentiment):
#     filtered_df = df[df['sentiment_category'] == sentiment]
#     if not filtered_df.empty:
#         return filtered_df['User Name'].value_counts().idxmax()
#     else:
#         return None
#
# most_complaints_user = most_tweets_by_sentiment('complaint')
# print("User with most complaints tweets:", most_complaints_user)
def extract_violation_types(df):
    def extract_violation_type(text):
        # Similar to extracting locations, extract violation types based on keywords
        # For simplicity, let's assume a few keywords representing violation types
        violation_keywords = [
            "no parking", "speeding", "causing accident", "illegal parking", "reckless driving",
            "road blockage", "signal violation", "overtaking", "wrong lane", "no helmet",
            "no pillion helmet", "triples riding", "road rage", "racing", "no number plate",
            "front bumper", "defective number plate", "wrong turn", "illegal entry", "no way",
            "possibility of accident", "causing nuisance", "pillion not wearing helmet", "not wearing helmet",
            "wrong side driving", "wrong driving", "causing obstruction"
        ]
        text_lower = text.lower()
        for keyword in violation_keywords:
            if keyword in text_lower:
                return keyword.capitalize()  # Capitalize the extracted violation type
        return "none"  # Return "unknown" only after checking all violation keywords

    # Apply the function to extract violation types from the 'translated_text' column
    df['Violation Type'] = df['translated_text'].apply(extract_violation_type)

    return df
extract_violation_types(df)
def extract_vehicle_number_from_texts(df):
    def extract_vehicle_number_from_text(text):
        # Regular expression pattern to match vehicle numbers
        pattern = r'\b[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,2}\s?[0-9]{1,4}\b'

        # Search for vehicle numbers in the text using the pattern
        matches = re.findall(pattern, text)

        if not matches:
            return "None"

        # Return the list of matched vehicle numbers
        return matches

    # Apply the function to extract vehicle numbers from the 'translated_text' column
    df['Vehicle Numbers'] = df['translated_text'].apply(extract_vehicle_number_from_text)
    return df
extract_vehicle_number_from_texts(df)
def extract_locations(df):
    def extract_location(text):
        # Use a list of location keywords
        chennai_location_keywords = [
            "Adambakkam", "Adyar", "Alandur", "Alapakkam", "Alwarpet", "Alwarthirunagar", "Ambattur",
            "Aminjikarai", "Anna Nagar", "Annanur", "Arumbakkam", "Ashok Nagar", "Avadi", "Ayanavaram",
            "Beemannapettai", "Besant Nagar", "Basin Bridge", "Chepauk", "Chetput", "Chintadripet",
            "Chitlapakkam", "Choolai", "Choolaimedu", "Chrompet", "Egmore", "Ekkaduthangal", "Eranavur",
            "Ennore", "Foreshore Estate", "Fort St. George", "George Town", "Gopalapuram", "Government Estate",
            "Guindy", "Guduvancheri", "IIT Madras", "Injambakkam", "ICF", "Iyyapanthangal", "Jafferkhanpet",
            "Kadambathur", "Karapakkam", "Kattivakkam", "Kattupakkam", "Kazhipattur", "K.K. Nagar",
            "Keelkattalai", "Kilpauk", "Kodambakkam", "Kodungaiyur", "Kolathur", "Korattur", "Korukkupet",
            "Kottivakkam", "Kotturpuram", "Kottur", "Kovilambakkam", "Koyambedu", "Kundrathur", "Madhavaram",
            "Madhavaram Milk Colony", "Madipakkam", "Madambakkam", "Maduravoyal", "Manali", "Manali New Town",
            "Manapakkam", "Mandaveli", "Mangadu", "Mannady", "Mathur", "Medavakkam", "Meenambakkam", "MGR Nagar",
            "Minjur", "Mogappair", "MKB Nagar", "Mount Road", "Moolakadai", "Moulivakkam", "Mugalivakkam",
            "Mudichur", "Mylapore", "Nandanam", "Nanganallur", "Nanmangalam", "Neelankarai", "Nemilichery",
            "Nesapakkam", "Nolambur", "Noombal", "Nungambakkam", "Otteri", "Padi", "Pakkam", "Palavakkam",
            "Pallavaram", "Pallikaranai", "Pammal", "Park Town", "Parry's Corner", "Pattabiram", "Pattaravakkam",
            "Pazhavanthangal", "Peerkankaranai", "Perambur", "Peravallur", "Perumbakkam", "Perungalathur",
            "Perungudi", "Pozhichalur", "Poonamallee", "Porur", "Pudupet", "Pulianthope", "Purasaiwalkam",
            "Puthagaram", "Puzhal", "Puzhuthivakkam/ Ullagaram", "Raj Bhavan", "Ramavaram", "Red Hills",
            "Royapettah", "Royapuram", "Saidapet", "Saligramam", "Santhome", "Sembakkam", "Selaiyur",
            "Shenoy Nagar", "Sholavaram", "Sholinganallur", "Sithalapakkam", "Sowcarpet", "St.Thomas Mount",
            "Surapet", "Tambaram", "Teynampet", "Tharamani", "T. Nagar", "Thirumangalam", "Thirumullaivoyal",
            "Thiruneermalai", "Thiruninravur", "Thiruvanmiyur", "Thiruvallur", "Tiruverkadu", "Thiruvotriyur",
            "Thuraipakkam", "Tirusulam", "Tiruvallikeni", "Tondiarpet", "United India Colony", "Vandalur",
            "Vadapalani", "Valasaravakkam", "Vallalar Nagar", "Vanagaram", "Velachery", "Velappanchavadi",
            "Villivakkam", "Virugambakkam", "Vyasarpadi", "Washermanpet", "West Mambalam",
            # Landmarks and Metro Stations
            "Chennai Central", "Marina Beach", "Kapaleeshwarar Temple", "Guindy National Park",
            "Arignar Anna Zoological Park", "Valluvar Kottam", "Chennai Egmore", "Chennai Central Metro Station",
            "Vadapalani Metro Station", "Anna Nagar Tower Metro Station", "St. Thomas Mount Metro Station",
            "Guindy Metro Station", "Koyambedu Metro Station", "Anna Salai", "Nungambakkam High Road", "Teynampet", "AG DMS Metro", "Gemini flyover",
            "Sterling Road", "Loyola College bridge", "Thirumangalam", "Saidapet",
            "Adambakkam", "Adyar", "Alandur", "Alapakkam", "Alwarpet", "Alwarthirunagar", "Ambattur",
            "Aminjikarai", "Anna Nagar", "Annanur", "Arumbakkam", "Ashok Nagar", "Avadi", "Ayanavaram",
            "Beemannapettai", "Besant Nagar", "Basin Bridge", "Chepauk", "Chetput", "Chintadripet",
            "Chitlapakkam", "Choolai", "Choolaimedu", "Chrompet", "Egmore", "Ekkaduthangal", "Eranavur",
            "Ennore", "Foreshore Estate", "Fort St. George", "George Town", "Gopalapuram", "Government Estate",
            "Guindy", "Guduvancheri", "IIT Madras", "Injambakkam", "ICF", "Iyyapanthangal", "Jafferkhanpet",
            "Kadambathur", "Karapakkam", "Kattivakkam", "Kattupakkam", "Kazhipattur", "K.K. Nagar",
            "Keelkattalai", "Kilpauk", "Kodambakkam", "Kodungaiyur", "Kolathur", "Korattur", "Korukkupet",
            "Kottivakkam", "Kotturpuram", "Kottur", "Kovilambakkam", "Koyambedu", "Kundrathur", "Madhavaram",
            "Madhavaram Milk Colony", "Madipakkam", "Madambakkam", "Maduravoyal", "Manali", "Manali New Town",
            "Manapakkam", "Mandaveli", "Mangadu", "Mannady", "Mathur", "Medavakkam", "Meenambakkam",
            "MGR Nagar", "Minjur", "Mogappair", "MKB Nagar", "Mount Road", "Moolakadai", "Moulivakkam",
            "Mugalivakkam", "Mudichur", "Mylapore", "Nandanam", "Nanganallur", "Nanmangalam", "Neelankarai",
            "Nemilichery", "Nesapakkam", "Nolambur", "Noombal", "Nungambakkam", "Otteri", "Padi", "Pakkam",
            "Palavakkam", "Pallavaram", "Pallikaranai", "Pammal", "Park Town", "Parry's Corner", "Pattabiram",
            "Pattaravakkam", "Pazhavanthangal", "Peerkankaranai", "Perambur", "Peravallur", "Perumbakkam",
            "Perungalathur", "Perungudi", "Pozhichalur", "Poonamallee", "Porur", "Pudupet", "Pulianthope",
            "Purasaiwalkam", "Puthagaram", "Puzhal", "Puzhuthivakkam/ Ullagaram", "Raj Bhavan", "Ramavaram",
            "Red Hills", "Royapettah", "Royapuram", "Saidapet", "Saligramam", "Santhome", "Sembakkam", "Selaiyur",
            "Shenoy Nagar", "Sholavaram", "Sholinganallur", "Sithalapakkam", "Sowcarpet", "St.Thomas Mount",
            "Surapet", "Tambaram", "Teynampet", "Tharamani", "T. Nagar", "Thirumangalam", "Thirumullaivoyal",
            "Thiruneermalai", "Thiruninravur", "Thiruvanmiyur", "Thiruvallur", "Tiruverkadu", "Thiruvotriyur",
            "Thuraipakkam", "Tirusulam", "Tiruvallikeni", "Tondiarpet", "United India Colony", "Vandalur",
            "Vadapalani", "Valasaravakkam", "Vallalar Nagar", "Vanagaram", "Velachery", "Velappanchavadi",
            "Villivakkam", "Virugambakkam", "Vyasarpadi", "Washermanpet", "West Mambalam",
            "Chennai Central", "Marina Beach", "Kapaleeshwarar Temple", "Guindy National Park",
            "Arignar Anna Zoological Park", "Valluvar Kottam", "Chennai Egmore", "Chennai Central Metro Station",
            "Vadapalani Metro Station", "Anna Nagar Tower Metro Station", "St. Thomas Mount Metro Station",
            "Guindy Metro Station", "Koyambedu Metro Station", "Sterling Road Signal", "Triplicane"
        ]
        # Remove duplicates
        chennai_location_keywords = list(set(chennai_location_keywords))

        # Use keywords to extract locations
        pattern = r'\b(?:' + '|'.join(chennai_location_keywords) + r')\b'

        # Search for location keywords in the text
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).capitalize()  # Capitalize the extracted location
        else:
            return None

    # Apply the function to extract locations from the 'translated_text' column
    df['Location'] = df['translated_text'].apply(extract_location)

    return df
extract_locations(df)
def save_sentiment_to_excel(df, folder_path, file_name="demochk.xlsx"):
    file = file_name
    excel_sentiment = os.path.join(folder_path, file)
    counts_df = pd.DataFrame({
        'Received Complaints': [received_complaints_count],
        'Resolved Complaints': [resolved_complaints_count]
    })
    # Concatenate the counts DataFrame with the scraped data DataFrame
    df = pd.concat([df, counts_df], axis=1)
    print(f"Saving sentiment analysis results to: {excel_sentiment}")
    df.to_excel(excel_sentiment, index=False)
    print("File saved successfully.")
save_sentiment_to_excel(df, folder_path)
def create_visualizations(df, folder_path, received_complaints_count, resolved_complaints_count):
    colors = {
        'positive': 'green',
        'negative': 'red',
        'unknown': 'blue',
        'complaint': 'orange'
    }

    # Sentiment Counts Bar Plot
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=[colors.get(cat, 'grey') for cat in sentiment_counts.index])
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'sentiment_distribution_bar.png'))
    plt.close()

    # Sentiment Counts Pie Chart
    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=[colors.get(cat, 'grey') for cat in sentiment_counts.index])
    plt.title('Distribution of Sentiment Categories')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'sentiment_distribution_pie.png'))
    plt.close()

    # Plot the time series of sentiment distribution over time
    df['Tweet Date'] = pd.to_datetime(df['Tweet Date'])
    df.set_index('Tweet Date', inplace=True)

    # Line Plot for Sentiment Distribution Over Time
    sentiment_counts_over_time = df.groupby([pd.Grouper(freq='D'), 'sentiment']).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    for category in sentiment_counts_over_time.columns:
        plt.plot(sentiment_counts_over_time.index, sentiment_counts_over_time[category], label=category.capitalize(), color=colors.get(category, 'grey'))
    plt.title('Sentiment Distribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend(title='Sentiment Category')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'sentiment_distribution_over_time.png'))
    plt.close()

    # Plot the received and resolved complaints counts
    plt.figure(figsize=(8, 6))
    plt.bar(['Acknowleged Complaints', 'Action-taken Complaints'], [received_complaints_count, resolved_complaints_count], color=['blue', 'red'])
    plt.title('Acknowleged vs Action-taken')
    plt.xlabel('Complaints Type')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'Acknowleged vs Action_Taken.png'))
    plt.close()
create_visualizations(df, folder_path, received_complaints_count, resolved_complaints_count)