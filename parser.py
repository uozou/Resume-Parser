
import cv2
import pytesseract
from google.colab.patches import cv2_imshow
import os
import PyPDF2
import matplotlib.pyplot as plt

file_path = "/content/minimalist-resume-template.png"
ext = os.path.splitext(file_path)[1]

if ext in ['.jpg', '.png', '.bmp']:
  # Load the image using OpenCV
  image = cv2.imread(file_path)
  # Convert BGR to RGB to work with other library
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Display the image
  plt.imshow(image)
  plt.show()

  # Apply OCR using Tesseract
  text = pytesseract.image_to_string(image, lang='eng')
  print('image')

elif ext in ['.txt']:
  #Read in the text report
  with open(file_path, 'r') as f:
    text = f.read()
    print('text')

elif ext in['.pdf']:
  # Open a PDF file and extract its text content
  with open(file_path, 'rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    # do something with the text

else:
    print('unknown format resume')

from matplotlib import pyplot as plt


# Print the extracted text
print(text)


import spacy
import re

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Apply named entity recognition
doc = nlp(text)

# Extract names
names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

# Remove duplicates
names = list(set(names))

# Print the names
print(names)

# Define a regex pattern to match email addresses
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Search for the pattern in the text using the re module
match = re.search(pattern, text)

# If a match is found, print the email address
if match:
    email = match.group()
    print(email)
else:
    print("No email address found.")

    
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
# Tokenize the text
tokens = word_tokenize(text)
for i in tokens:
  if len(i)<=2:
    tokens.remove(i)
print(tokens)
# Remove stop words
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token.lower() not in stop_words]

# Extract key words commonly occuring as it gives idea about strength of candidate
word_counts = Counter(tokens)
num_keywords = 5 # Change this to the number of key words you want to extract
keywords = [word for word, count in word_counts.most_common(num_keywords)]

# Print the key words
print(keywords)

# Define a list of skill-related words
skill_words = ['aptitude','python', 'drawing' ,'programming', 'coding','Bank','focused','finance', 'data analysis', 'machine learning', 'statistics']

# Extract key words related to skills
skill_tokens = [token for token in tokens if token.lower() in skill_words]

# Remove duplicates
skill_tokens = list(set(skill_tokens))

# Print the skill tokens
print(skill_tokens)

import sqlite3

# Create a connection to the database
conn = sqlite3.connect('resume.db')

# Create a cursor object
cur = conn.cursor()

# Define the SQL query to create a new table
sql_query = '''CREATE TABLE user (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    skills TEXT
)'''

# Execute the SQL query to create the table
cur.execute(sql_query)

# Define the data to be inserted
data = (1,  names[0], email, str(skill_tokens))

# Define the SQL query to insert data into the table
sql_query = 'INSERT INTO user (id, name, email, skills) VALUES (?, ?, ?, ?)'

# Execute the SQL query with the data
cur.execute(sql_query, data)

# Commit the changes to the database
conn.commit()

# Close the connection
conn.close()



