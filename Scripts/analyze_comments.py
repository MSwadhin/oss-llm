import json
import argparse
from utils.gh_api_manager import GH_API_Manager
import sys
import time
import datetime
import base64

import tree_sitter_javascript as jssitter
import tree_sitter_python as pysitter
from tree_sitter import Language, Parser


# add argument
parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str,required=True)


args = parser.parse_args()
config_file = args.config


with open(config_file,"r") as f:
    config = json.load(f)

gh_api_manager = GH_API_Manager(config['gh_tokens'])


ds_path = config["full_dataset_path"]
query_file = config["query_path"]
input_file = ds_path + "/ds_repos.json"
output_file = ds_path + "/ds_repos_comments_filtered.json"
lang = config['repo-search']['languages'][0]

print(lang)

if lang=="JavaScript":
    LANGUAGE = Language(jssitter.language())
elif lang=="Python":
    LANGUAGE = Language(pysitter.language())

parser = Parser(LANGUAGE)


with open(input_file,"r") as f:
    DATA = json.load(f)

with open(query_file,"r") as f:
    QUERIES = json.load(f)



def extract_comments_with_line_numbers(source_code):
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    lines = source_code.splitlines()
    comments = []
    def traverse(node):
        if node.type == "comment":
            comment_text = source_code[node.start_byte:node.end_byte]
            line_number = node.start_point[0] + 1  # 0-based index
            comments.append((line_number, comment_text.strip()))
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return comments


def process_code(code):
    matches = []
    comments = extract_comments_with_line_numbers(code)
    for query in QUERIES:
        words = query.split(" ")
        pattern = " ".join(words[:-1])
        pattern = pattern.replace("\"", "")
        pattern = pattern.lower()
        for ln,comment in comments:
            text = comment.lower()
            if pattern in text:
                d = {
                    "comment": comment,
                    "line": ln,
                    "pattern": pattern
                }
                matches.append(d)
    return matches



def process_repo(repo_name):

    print("Processing Repo : {repo_name}")

    indx = 0
    for file in DATA[repo_name]['files']:
        url = file['url']
        resp = gh_api_manager.make_request(url)
        time.sleep(0.3)
        sys.stdout.flush()
        if not resp:
            print(f"Error fetching code for {url}")
            indx += 1
            continue
        
        if not "content" in resp:
            print(f"Error fetching content for {url}")
            indx += 1
            continue
        content = resp["content"]
        content = base64.b64decode(content).decode("utf-8")
        matches = process_code(content)
        file['matches'] = matches
        DATA[repo_name]['files'][indx] = file
        indx += 1
        print(f"found for file {file['name']} :::\n {matches}")


if __name__ == "__main__":

    counter = 0
    for key,value in DATA.items():
        process_repo(key)
        counter += 1
        if counter % 100 == 0:
            with open(output_file,"w") as f:
                json.dump(DATA,f)
    

    with open(output_file,"w") as f:
        json.dump(DATA,f)