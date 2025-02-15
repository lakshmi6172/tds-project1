import subprocess
from datetime import datetime
from dateutil.parser import parse
import json
import os
import requests
import re
import base64
import numpy as np
import sqlite3
from scipy.spatial.distance import cosine
from resp import url,headers,token
def create_data(path,email):
    print(path)
    subprocess.run(['uv','run',path,email],capture_output=True, text=True, check=True)  
    
def format_data(path,version):
    file_path ='.'+path  
    subprocess.run('npx prettier@{version} --write {path}'.format(version=version,path=file_path),shell=True,check=True)
    original=open(file_path).read()
    expected = subprocess.run(
        ["npx", "prettier@3.4.2", "--stdin-filepath", path],
        input=original,
        capture_output=True,
        text=True,
        check=True,
        shell=True,
    ).stdout


def count_weekday_occurrences(input_file, output_file, target_day):
    # Map weekday names to their corresponding integer values (Monday=0, ..., Sunday=6)
    input_file = os.path.abspath('.'+input_file)
    output_file = os.path.abspath('.'+output_file)
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }
    day_code=weekday_map[target_day.lower()]
    # Count the occurrences of the target weekday in the input file
    count = 0
    dates=open(input_file).read().splitlines()
    ct=sum(1 for date in dates if parse(date).weekday() == day_code)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(ct))

def sort_array(input_file,first_target,second_target,output_file):
    input_file = os.path.abspath('.'+input_file)
    output_file = os.path.abspath('.'+output_file)
    contacts=json.load(open(input_file))
    contacts.sort(key=lambda x: (x[first_target], x[second_target]))
    with open(output_file, "w") as f:
        json.dump(contacts, f, indent=4)

def write_recent_first_lines(input_dir, output_file, num_files):
    input_dir = os.path.abspath('.'+input_dir)
    output_file = os.path.abspath('.'+output_file)
    all_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    all_files.sort(key=os.path.getmtime, reverse=True)
    first_lines = []
    for file_path in all_files[:num_files]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                first_line = file.readline().rstrip('\n')
                first_lines.append(first_line)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    try:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            expected = "".join([f+"\n" for f in first_lines])
            out_file.write(expected)
    except Exception as e:
        print(f"Error writing to file {output_file}: {e}")

def extract_sender_email(inputfile, outputfile):
    inputfile = os.path.abspath('.'+inputfile)
    outputfile = os.path.abspath('.'+outputfile)
    s=open(inputfile).read()
    req=requests.post(url=url,headers=headers,json={
     "model":"gpt-4o-mini",
    "messages":[
        {"role": "system", "content": "Extract the sender's email address from the given email text. Reply with only the email address and nothing else."},
        {"role": "user", "content": s}
    ],})
    email=req.json()['choices'][0]['message']['content']
    with open(outputfile, 'w', encoding='utf-8') as out_file:
        out_file.write(email)

def generate_markdown_index(directory, output_file):
    index = {}
    output_file='.'+output_file
    directory='.'+directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):  # Only process markdown files
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = re.match(r"^#\s+(.+)", line)  # Find first H1 heading
                        if match:
                            index[relative_path] = (match.group()).lstrip("# ")
                            break  # Stop after first H1
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

import base64
def png_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


def card_ocr( image_path='/data/credit_card.png',filename='/data/credit_card.txt'):
    AIPROXY_TOKEN = token
    image_path = '.' + image_path
    filename = '.' + filename
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "There is 8 or more digit number is there in this image, with space after every 4 digit, only extract the those digit number without spaces and return just the number without any other characters"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_to_base64(image_path)}"
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    # Make the request to the AIProxy service
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                             headers=headers, data=json.dumps(body))
    # response.raise_for_status()

    # Extract the credit card number from the response
    result = response.json()
    # print(result); return None
    card_number = result['choices'][0]['message']['content'].replace(" ", "")

    # Write the extracted card number to the output file
    with open(filename, 'w') as file:
        file.write(card_number)
# A8()




def get_embedding(text):
    AIPROXY_TOKEN = token
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": [text]
    }
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def similar_comments(filename='/data/comments.txt', output_filename='/data/comments-similar.txt'):
    filename = '.' + filename
    output_filename = '.' + output_filename
    # Read comments
    with open(filename, 'r') as f:
        comments = [line.strip() for line in f.readlines()]

    # Get embeddings for all comments
    embeddings = [get_embedding(comment) for comment in comments]

    # Find the most similar pair
    min_distance = float('inf')
    most_similar = (None, None)

    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            distance = cosine(embeddings[i], embeddings[j])
            if distance < min_distance:
                min_distance = distance
                most_similar = (comments[i], comments[j])

    # Write the most similar pair to file
    with open(output_filename, 'w') as f:
        f.write(most_similar[0] + '\n')
        f.write(most_similar[1] + '\n')
def sql_query(database,table,Type,outputfile):
    conn=sqlite3.connect(database)
    outputfile=os.path.abspath('.'+outputfile)
    cursor=conn.cursor()
    Type=Type.lower()
    q='select sum(price*units) from {table} where lower(trim(type))="{Type}"'.format(table=table,Type=Type)
    cursor.execute(q)
    result=cursor.fetchone()[0]
    with open(outputfile,"w",encoding="utf-8") as file:
        file.write(str(result))
    conn.close()
