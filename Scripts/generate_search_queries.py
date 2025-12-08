from utils.gh_api_manager import GH_API_Manager
from utils.es_manager import ESManager
import argparse
import json
import datetime
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
api_manager = GH_API_Manager(config['gh_tokens'])

repo_search_options = config['repo-search']


SEARCH_URL = "https://api.github.com/search/code"
QUERYS_FILE_PATH = config['query_path']


def generate_search_qeries():
    queries = []
    phraseA = repo_search_options["phrase-A"]
    phraseB = repo_search_options["phrase-B"]
    phraseC = repo_search_options["phrase-C"]
    languages = repo_search_options["languages"]

    # generate queries of the form: A * B
    for lang in languages:
        for phrase in phraseA:
            for phrase2 in phraseB:
                queries.append(f'"{phrase} {phrase2}" language:{lang}')
    
    # generate queries of the form: B * C * A
    for lang in languages:
        for phrase in phraseB:
            for phrase2 in phraseC:
                for phrase3 in phraseA:
                    queries.append(f'"{phrase} {phrase2} {phrase3}" language:{lang}')

    # filter queries that gives too less results
    min_result_count = repo_search_options["min_result_count"]
    filtered_queries = {}
    all_queries = {}
    for query in queries:
        params = {
            "q": query,
            "per_page": 1,  # max 100
            "page": 1
        }
        response = api_manager.make_request(SEARCH_URL, params=params)
        print(f"current time : {datetime.datetime.now()}", flush=True)
        time.sleep(4) # to avoid rate limit
        if not response:
            print("Error in response")
            break
        if not "items" in response or not "total_count" in response:
            print("Error in response")
            continue

        found_items = response.get('total_count', 0)
        print(f"Query: {query} ---- Found items: {found_items}")
        all_queries[query] = found_items
        
    
    with open(QUERYS_FILE_PATH, "w") as f:
        json.dump(all_queries, f, indent=4)
    
    


if __name__=='__main__':
    generate_search_qeries()
    

    
