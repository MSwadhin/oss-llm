import json
import pandas as pd
import argparse
from utils.gh_api_manager import GH_API_Manager
import sys
import time



# add argument
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)

# parse argument
# get config filename from --config arg parameter
args = parser.parse_args()
config_file_name = args.config

#open and load config.json
cf = open(config_file_name,'r')
config = json.load(cf)

gh_api_manager = GH_API_Manager(config['gh_tokens'])


QUERY_FILE_PATH = config['query_path']
OUTPUT_DIR = config['output_directory']
FULL_DS_PATH = config['full_dataset_path']


with open(QUERY_FILE_PATH, "r") as f:
    all_queries = json.load(f)
    queries = list(all_queries.keys())


REPO_DATA = {}



def get_contributors_count(repo_full_name):
    url = f"https://api.github.com/repos/{repo_full_name}/contributors"
    response = gh_api_manager.make_request(url)
    if not response:
        print(f"Error fetching contributors for {repo_full_name}")
        return 0
    contributors_count = len(response)
    return contributors_count

def fetch_repo_details(repo_full_name,name):


    url = f"https://api.github.com/repos/{repo_full_name}"
    response = gh_api_manager.make_request(url)
    print("start fetching repo details for " + repo_full_name)
    print("======================================================")
    if not response:
        print(f"Error fetching repo details for {repo_full_name}")
        return None
    star_count = response.get("stargazers_count", 0)
    fork_count = response.get("forks_count", 0)
    size_of_repo = response.get("size", 0)
    contributor_count = get_contributors_count(repo_full_name)
    cerated_at = response.get("created_at", "")
    updated_at = response.get("updated_at", "")
    pushed_at = response.get("pushed_at", "")
    # readme_found = readme_data.get(repo_full_name, {}).get("found", False)
    # if readme_found:
    #     readme_url = readme_data[repo_full_name]["html_url"]
    # else:
    #     readme_url = None
    REPO_DATA[name]["metrics"] = {
        "stars": star_count,
        "forks": fork_count,
        "size": size_of_repo,
        "contributors": contributor_count,
        "created_at": cerated_at,
        "updated_at": updated_at,
        "pushed_at": pushed_at,
        # "readme_found": readme_found,
        # "readme_url": readme_url,
        "code_file_found": len(REPO_DATA[name]["files"]),
    }

    print(f"Fetched details for {repo_full_name}:")
    print(REPO_DATA[name]['metrics'])
    print("======================================================", flush=True)
    time.sleep(0.20)



def process_code(code_data, query):
    repo = code_data['repository']['name']

    if repo not in REPO_DATA:
        REPO_DATA[repo] = {
            "repo_details" : code_data['repository'],
            "files": [],
            "total_count": 0,
            "queries": set()
        }
    
    REPO_DATA[repo]['files'].append(code_data)
    REPO_DATA[repo]['total_count'] += 1
    REPO_DATA[repo]['queries'].add(query)


def process_query(query):
    query = query.replace(" ", "_")
    file_path = f"{OUTPUT_DIR}/{query}.json"
    with open(file_path, "r") as f:
        code_data = json.load(f)
        for data in code_data:
            process_code(data,query=query)
        print(f"Processed {len(code_data)} items for query: {query}")

def run():
    for query in queries:
        if all_queries[query] < 1:
            continue
        process_query(query)
    
    print(f"Total repositories: {len(REPO_DATA)}")

    for key,data in REPO_DATA.items():
        data['queries'] = list(data['queries'])
        data['query_count'] = len(data['queries'])
    

    for key,data in REPO_DATA.items():
        fullname = data['repo_details']['full_name']
        fetch_repo_details(fullname,key)


    sorted_data = sorted(REPO_DATA.items(), key=lambda x: x[1]['total_count'], reverse=True)
    # REPO_DATA = {k: v for k, v in sorted_data}

    with open(f"{FULL_DS_PATH}/ds_repos.json", "w") as f:
        json.dump({k: v for k, v in sorted_data}, f, indent=4)
    
    print(f"Total repositories: {len(REPO_DATA)}")




if __name__=='__main__':
    run()
    

