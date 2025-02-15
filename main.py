import os
import subprocess
import json
import re
import datetime
import sqlite3
import hashlib
import base64
import shlex
import io
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
app = Flask(__name__)

CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins
     supports_credentials=True)  # Allow credentials

# Constants
DATA_DIR = "/data"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
LLM_MODEL = "gpt-4o-mini"  # Enforce GPT-4o-Mini

# Ensure the /data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Helper functions
def execute_command(command):
    """Executes a shell command and returns the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Command failed: {e.stderr}")

def read_file(path):
    """Reads the content of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def write_file(path, content):
    """Writes content to a file."""
    try:
        with open(path, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        raise Exception(f"Failed to write to file: {e}")

def call_llm(prompt):
    """Calls the LLM with the given prompt."""
    url = "https://api.tds.ai/generate"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": LLM_MODEL,
        "prompt": prompt
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)  # Reduced timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM call failed: {e}")

def is_safe_path(path):
    """Checks if the path is within the /data directory."""
    abs_path = os.path.abspath(path)
    abs_data_dir = os.path.abspath(DATA_DIR)
    return abs_path.startswith(abs_data_dir)

def enforce_security(task_description):
    """Enforces security policies: no access outside /data, no deletion."""
    if "delete" in task_description.lower():
        raise ValueError("Deletion of files is not allowed.")
    if "rm " in task_description.lower():
        raise ValueError("Deletion of files is not allowed.")
    if "access" in task_description.lower() and "outside /data" in task_description.lower():
        raise ValueError("Accessing files outside /data is not allowed.")
    if "exfiltrate" in task_description.lower():
        raise ValueError("Exfiltration of data is not allowed.")

# Task Handlers
def handle_a1(user_email):
    """Installs uv (if required) and runs datagen.py."""
    try:
        # Check if uv is installed
        try:
            subprocess.run(["uv", "--version"], check=False, capture_output=True)
        except FileNotFoundError:
            # Install uv if not found
            execute_command("pip install uv")

        datagen_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
        execute_command(f"python3 -c \"$(curl -fsSL {datagen_url})\" {user_email}")
        return "Data generation completed."
    except Exception as e:
        raise Exception(f"Task A1 failed: {e}")

def handle_a2():
    """Formats /data/format.md using prettier@3.4.2."""
    try:
        # Check if prettier is installed
        try:
            subprocess.run(["prettier", "--version"], check=False, capture_output=True)
        except FileNotFoundError:
            # Install prettier if not found
            execute_command("npm install -g prettier@3.4.2")

        execute_command(f"prettier --write {DATA_DIR}/format.md")
        return "File formatted with prettier."
    except Exception as e:
        raise Exception(f"Task A2 failed: {e}")

def handle_a3():
    """Counts Wednesdays in /data/dates.txt and writes the count to /data/dates-wednesdays.txt."""
    try:
        dates_content = read_file(f"{DATA_DIR}/dates.txt")
        if not dates_content:
            raise FileNotFoundError("dates.txt not found")

        wednesday_count = 0
        for line in dates_content.splitlines():
            try:
                date_obj = datetime.datetime.strptime(line.strip(), "%Y-%m-%d")
                if date_obj.weekday() == 2:  # Wednesday is 2
                    wednesday_count += 1
            except ValueError:
                pass  # Ignore invalid date formats

        write_file(f"{DATA_DIR}/dates-wednesdays.txt", str(wednesday_count))
        return "Wednesday count written to file."
    except Exception as e:
        raise Exception(f"Task A3 failed: {e}")
        
        
        from datetime import datetime

def handle_a4():
    """Sorts contacts in /data/contacts.json by last_name, then first_name."""
    try:
        contacts_path = f"{DATA_DIR}/contacts.json"
        contacts_content = read_file(contacts_path)
        if not contacts_content:
            raise FileNotFoundError("contacts.json not found")

        contacts = json.loads(contacts_content)
        sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
        write_file(f"{DATA_DIR}/contacts-sorted.json", json.dumps(sorted_contacts, indent=2))
        return "Contacts sorted and written to file."
    except Exception as e:
        raise Exception(f"Task A4 failed: {e}")

def handle_a5():
    """Writes the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt."""
    try:
        log_dir = f"{DATA_DIR}/logs/"
        if not os.path.exists(log_dir):
            raise FileNotFoundError("Logs directory not found")

        log_files = [f"{log_dir}{f}" for f in os.listdir(log_dir) if f.endswith(".log")]
        log_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
        recent_logs = log_files[:10]

        recent_lines = []
        for log_file in recent_logs:
            first_line = read_file(log_file).splitlines()[0] if read_file(log_file) else ""
            recent_lines.append(first_line)

        write_file(f"{DATA_DIR}/logs-recent.txt", "\n".join(recent_lines))
        return "First lines of recent logs written to file."
    except Exception as e:
        raise Exception(f"Task A5 failed: {e}")

def handle_a6():
    """Creates an index file /data/docs/index.json mapping filenames to H1 titles."""
    try:
        docs_dir = f"{DATA_DIR}/docs/"
        if not os.path.exists(docs_dir):
            raise FileNotFoundError("Docs directory not found")

        index = {}
        for filename in os.listdir(docs_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(docs_dir, filename)
                content = read_file(filepath)
                if content:
                    # Find the first H1 occurrence
                    match = re.search(r"^#\s+(.*)$", content, re.MULTILINE)
                    title = match.group(1) if match else ""
                    index[filename] = title

        write_file(f"{DATA_DIR}/docs/index.json", json.dumps(index, indent=2))
        return "Index file created."
    except Exception as e:
        raise Exception(f"Task A6 failed: {e}")

def handle_a7():
    """Extracts the sender's email address from /data/email.txt using an LLM."""
    try:
        email_content = read_file(f"{DATA_DIR}/email.txt")
        if not email_content:
            raise FileNotFoundError("email.txt not found")

        prompt = f"Extract the sender's email address from the following email:\n{email_content}\n\nSender's email address:"
        sender_email = call_llm(prompt)
        write_file(f"{DATA_DIR}/email-sender.txt", sender_email)
        return "Sender's email address extracted and written to file."
    except Exception as e:
        raise Exception(f"Task A7 failed: {e}")

def handle_a8():
    """Extracts the credit card number from /data/credit-card.png using an LLM."""
    try:
        image_path = f"{DATA_DIR}/credit-card.png"
        if not os.path.exists(image_path):
            raise FileNotFoundError("credit-card.png not found")

        # Encode the image to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = f"Extract the credit card number from the following image (base64 encoded):\n{encoded_string}\n\nCredit card number (without spaces):"
        card_number = call_llm(prompt)
        card_number = card_number.replace(" ", "")  # Remove spaces
        write_file(f"{DATA_DIR}/credit-card.txt", card_number)
        return "Credit card number extracted and written to file."
    except Exception as e:
        raise Exception(f"Task A8 failed: {e}")

def handle_a9():
    """Finds the most similar pair of comments in /data/comments.txt using embeddings."""
    try:
        comments_content = read_file(f"{DATA_DIR}/comments.txt")
        if not comments_content:
            raise FileNotFoundError("comments.txt not found")

        comments = [line.strip() for line in comments_content.splitlines() if line.strip()]

        if len(comments) < 2:
            raise ValueError("Not enough comments to compare.")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(comments)

        # Compute cosine similarity between all pairs
        similarities = util.cos_sim(embeddings, embeddings)

        # Find the pair with the highest similarity (excluding self-similarity)
        max_similarity = -1
        best_pair = None
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                if similarities[i][j] > max_similarity:
                    max_similarity = similarities[i][j]
                    best_pair = (comments[i], comments[j])

        if best_pair:
            write_file(f"{DATA_DIR}/comments-similar.txt", f"{best_pair[0]}\n{best_pair[1]}")
            return "Most similar comments written to file."
        else:
            raise Exception("Could not find a similar pair of comments.")
    except Exception as e:
        raise Exception(f"Task A9 failed: {e}")

def handle_a10():
    """Calculates the total sales of "Gold" tickets in /data/ticket-sales.db."""
    try:
        db_path = f"{DATA_DIR}/ticket-sales.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError("ticket-sales.db not found")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        result = cursor.fetchone()[0]

        total_sales = result if result else 0  # Handle cases where there are no Gold tickets

        write_file(f"{DATA_DIR}/ticket-sales-gold.txt", str(total_sales))
        conn.close()
        return "Total sales of Gold tickets written to file."
    except Exception as e:
        raise Exception(f"Task A10 failed: {e}")

def handle_b3(api_url, file_path):
    """Fetches data from an API and saves it to a file."""
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        write_file(file_path, response.text)
        return f"Data fetched from {api_url} and saved to {file_path}."
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")
    except Exception as e:
        raise Exception(f"Task B3 failed: {e}")

def handle_b4(repo_url, commit_message, file_path, file_content):
    """Clones a git repo, makes a commit, and pushes the changes."""
    try:
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(DATA_DIR, repo_name)

        # Clone the repository
        execute_command(f"git clone {repo_url} {repo_path}")

        # Write the file content to the specified path within the repo
        full_file_path = os.path.join(repo_path, file_path)
        write_file(full_file_path, file_content)

        # Add, commit, and push the changes
        execute_command(f"cd {repo_path} && git add .")
        execute_command(f"cd {repo_path} && git commit -m '{commit_message}'")
        execute_command(f"cd {repo_path} && git push")

        return f"Repository cloned, file added, committed, and pushed to {repo_url}."
    except Exception as e:
        raise Exception(f"Task B4 failed: {e}")

def handle_b5(db_type, db_path, query, output_path):
    """Runs a SQL query on a SQLite or DuckDB database and writes the result to a file."""
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"{db_type} database not found at {db_path}")

        if db_type.lower() == "sqlite":
            conn = sqlite3.connect(db_path)
        elif db_type.lower() == "duckdb":
            import duckdb
            conn = duckdb.connect(db_path)
        else:
            raise ValueError("Unsupported database type.  Must be 'sqlite' or 'duckdb'.")

        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        # Format the result as a string
        output = "\n".join([str(row) for row in result])

        write_file(output_path, output)
        conn.close()
        return f"SQL query executed on {db_type} database and result written to {output_path}."
    except Exception as e:
        raise Exception(f"Task B5 failed: {e}")

def handle_b6(url, selector, output_path):
    """Extracts data from a website using CSS selectors and saves it to a file."""
    try:
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        elements = soup.select(selector)
        extracted_data = "\n".join([element.text.strip() for element in elements])

        write_file(output_path, extracted_data)
        return f"Data scraped from {url} using selector '{selector}' and saved to {output_path}."
    except Exception as e:
        raise Exception(f"Task B6 failed: {e}")

def handle_b7(image_path, output_path, new_width=None, new_height=None, quality=85):
    """Compresses or resizes an image."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = Image.open(image_path)

        if new_width and new_height:
            img = img.resize((int(new_width), int(new_height)))
        elif new_width:
            width_percent = (int(new_width) / float(img.size[0]))
            height_size = int((float(img.size[1]) * float(width_percent)))
            img = img.resize((int(new_width), height_size))
        elif new_height:
            height_percent = (int(new_height) / float(img.size[1]))
            width_size = int((float(img.size[0]) * float(height_percent)))
            img = img.resize((width_size, int(new_height)))

        img.save(output_path, optimize=True, quality=quality)
        return f"Image compressed/resized and saved to {output_path}."
    except Exception as e:
        raise Exception(f"Task B7 failed: {e}")

def handle_b8(mp3_path, output_path):
    """Transcribes audio from an MP3 file."""
    try:
        import speech_recognition as sr

        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 file not found at {mp3_path}")

        r = sr.Recognizer()
        with sr.AudioFile(mp3_path) as source:
            audio = r.record(source)

        try:
            text = r.recognize_google(audio)
            write_file(output_path, text)
            return f"Audio transcribed and saved to {output_path}."
        except sr.UnknownValueError:
            raise Exception("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            raise Exception(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        raise Exception(f"Task B8 failed: {e}")

def handle_b9(markdown_path, output_path):
    """Converts Markdown to HTML."""
    try:
        import markdown

        markdown_content = read_file(markdown_path)
        if not markdown_content:
            raise FileNotFoundError(f"Markdown file not found at {markdown_path}")

        html_content = markdown.markdown(markdown_content)
        write_file(output_path, html_content)
        return f"Markdown converted to HTML and saved to {output_path}."
    except Exception as e:
        raise Exception(f"Task B9 failed: {e}")

def handle_b10(csv_path, filter_column, filter_value, output_path):
    """Writes an API endpoint that filters a CSV file and returns JSON data."""
    try:
        import pandas as pd

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        df = pd.read_csv(csv_path)

        if filter_column not in df.columns:
            raise ValueError(f"Filter column '{filter_column}' not found in CSV file.")

        filtered_df = df[df[filter_column] == filter_value]
        json_data = filtered_df.to_json(orient="records")

        write_file(output_path, json_data)
        return f"Filtered CSV data converted to JSON and saved to {output_path}."
    except Exception as e:
        raise Exception(f"Task B10 failed: {e}")

# Main API Endpoints
@app.route("/run", methods=["POST"])
def run_task():
    """Executes a task based on the provided description."""
    task_description = request.args.get("task")
    if not task_description:
        return jsonify({"error": "Task description is required."}), 400

    try:
        enforce_security(task_description)

        # LLM-powered task parsing and execution
        prompt = f"You are an autonomous agent that can execute tasks on a Linux system.  You have access to the following functions:\n"
        prompt += "- execute_command(command): Executes a shell command and returns the output.\n"
        prompt += "- read_file(path): Reads the content of a file.\n"
        prompt += "- write_file(path, content): Writes content to a file.\n"
        prompt += "You must use these functions to accomplish the task.  All file paths must be within the /data directory.  Do not attempt to delete any files.\n"
        prompt += f"Task: {task_description}\n"
        prompt += "Instructions: Provide a sequence of function calls to accomplish the task.  Be as concise as possible.  If the task is ambiguous, make reasonable assumptions.\n"
        prompt += "Example:\n"
        prompt += "Task: Count the number of lines in /data/file.txt and write the number to /data/count.txt\n"
        prompt += "Response:\n"
        prompt += "execute_command('wc -l /data/file.txt | awk \'{print $1}\'')\n"
        prompt += "write_file('/data/count.txt', execute_command('wc -l /data/file.txt | awk \'{print $1}\''))\n"
        prompt += "Now, respond to the following task:\n"
        prompt += f"Task: {task_description}\n"
        prompt += "Response:\n"

        llm_response = call_llm(prompt)

        # Execute the steps from the LLM response
        try:
            # Split the LLM response into individual commands
            commands = llm_response.split("\n")

            # Execute each command
            for command in commands:
                # Extract the function name and arguments from the command
                match = re.match(r"(\w+)\((.*)\)", command)
                if match:
                    function_name = match.group(1)
                    arguments_string = match.group(2)

                    # Parse the arguments string into a list of arguments
                    #arguments = [arg.strip().strip("'").strip('"') for arg in re.split(r",(?=(?:[^\'"]*[\'"]){2})*[^'""]*$)", arguments_string)]
                    arguments = [arg.strip() for arg in shlex.split(arguments_string)]
                    # Call the appropriate function based on the function name
                    if function_name == "execute_command":
                        if len(arguments) != 1:
                            raise ValueError("execute_command requires exactly one argument.")
                        execute_command(arguments[0])
                    elif function_name == "read_file":
                        if len(arguments) != 1:
                            raise ValueError("read_file requires exactly one argument.")
                        read_file(arguments[0])
                    elif function_name == "write_file":
                        if len(arguments) != 2:
                            raise ValueError("write_file requires exactly two arguments.")
                        write_file(arguments[0], arguments[1])
                    else:
                        raise ValueError(f"Unknown function: {function_name}")
        except Exception as e:
            # If the LLM response is not valid Python code, try to handle the task directly
            if "install uv" in task_description.lower() and "datagen.py" in task_description.lower():
                user_email = re.search(r"[\w\.-]+@[\w\.-]+", task_description).group(0)
                handle_a1(user_email)
            elif "format" in task_description.lower() and "/data/format.md" in task_description.lower():
                handle_a2()
            elif "wednesday" in task_description.lower() and "/data/dates.txt" in task_description.lower():
                handle_a3()
            elif "sort" in task_description.lower() and "/data/contacts.json" in task_description.lower():
                handle_a4()
            elif "most recent .log file" in task_description.lower() and "/data/logs/" in task_description.lower():
                handle_a5()
            elif "markdown" in task_description.lower() and "/data/docs/" in task_description.lower():
                handle_a6()
            elif "email" in task_description.lower() and "/data/email.txt" in task_description.lower():
                handle_a7()
            elif "credit card" in task_description.lower() and "/data/credit-card.png" in task_description.lower():
                handle_a8()
            elif "comments" in task_description.lower() and "/data/comments.txt" in task_description.lower():
                handle_a9()
            elif "ticket-sales.db" in task_description.lower():
                handle_a10()
            elif "fetch data from an api" in task_description.lower():
                api_url = re.search(r"https?://[^\s]+", task_description).group(0)
                file_path = re.search(r"/data/[^\s]+", task_description).group(0)
                handle_b3(api_url, file_path)
            elif "clone a git repo" in task_description.lower():
                repo_url = re.search(r"https?://[^\s]+", task_description).group(0)
                file_path = re.search(r"/data/[^\s]+", task_description).group(0)
                file_content = "Initial commit"
                commit_message = "Initial commit"
                handle_b4(repo_url, commit_message, file_path, file_content)
            elif "run a sql query" in task_description.lower():
                db_type = re.search(r"(sqlite|duckdb)", task_description, re.IGNORECASE).group(0)
                db_path = re.search(r"/data/[^\s]+", task_description).group(0)
                query = re.search(r"SELECT.*", task_description, re.IGNORECASE).group(0)
                output_path = re.search(r"/data/[^\s]+", task_description[task_description.find(query):]).group(0)
                handle_b5(db_type, db_path, query, output_path)
            elif "extract data from" in task_description.lower() or "scrape" in task_description.lower():
                url = re.search(r"https?://[^\s]+", task_description).group(0)
                selector = re.search(r"'(.*?)'", task_description).group(1)
                output_path = re.search(r"/data/[^\s]+", task_description).group(0)
                handle_b6(url, selector, output_path)
            elif "compress" in task_description.lower() or "resize" in task_description.lower():
                image_path = re.search(r"/data/[^\s]+\.(png|jpg|jpeg)", task_description).group(0)
                output_path = re.search(r"/data/[^\s]+\.(png|jpg|jpeg)", task_description[task_description.find(image_path)+1:]).group(0)
                handle_b7(image_path, output_path)
            elif "transcribe audio" in task_description.lower():
                mp3_path = re.search(r"/data/[^\s]+\.mp3", task_description).group(0)
                output_path = re.search(r"/data/[^\s]+", task_description[task_description.find(mp3_path)+1:]).group(0)
                handle_b8(mp3_path, output_path)
            elif "convert markdown to html" in task_description.lower():
                markdown_path = re.search(r"/data/[^\s]+\.md", task_description).group(0)
                output_path = re.search(r"/data/[^\s]+\.html", task_description).group(0)
                handle_b9(markdown_path, output_path)
            elif "filter a csv file" in task_description.lower():
                csv_path = re.search(r"/data/[^\s]+\.csv", task_description).group(0)
                filter_column = re.search(r"filter column '(.*?)'", task_description).group(1)
                filter_value = re.search(r"filter value '(.*?)'", task_description).group(1)
                output_path = re.search(r"/data/[^\s]+\.json", task_description).group(0)
                handle_b10(csv_path, filter_column, filter_value, output_path)
            else:
                raise e

        return jsonify({"message": "Task completed successfully."}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/read", methods=["GET"])
def read_file_content():
    """Returns the content of the specified file."""
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "File path is required."}), 400

    if not is_safe_path(path):
        return jsonify({"error": "Accessing files outside /data is not allowed."}), 400

    content = read_file(path)
    if content is None:
        return jsonify({"error": "File not found."}), 404
    return content, 200, {"Content-Type": "text/plain"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)