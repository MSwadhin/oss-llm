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

QUERIES_FILE_PATH = config['query_path']
OUTPUT_DIR = config['output_directory']


def get_code_search_results():


    min_result_count = repo_search_options["min_result_count"]
    with open(QUERIES_FILE_PATH, "r") as f:
        all_queries = json.load(f)
        queries = list(all_queries.keys())


    for query in queries:
        params = {
            "q": query,
            "per_page": 100,  # max 100
            "page": 1
        }
        if all_queries[query] < 1:
            continue
        page = 0
        cur_results = []
        while True:
            params['page'] = page
            response = api_manager.make_request(SEARCH_URL, params=params)
            print(f"current time : {datetime.datetime.now()}", flush=True)
            print(f"current query : {query}, page : {params['page']}", flush=True)
            time.sleep(4)
            page += 1
            if not response:
                break
            if not "items" in response:
                break
            items = response['items']
            if(len(items) < 1):
                break

            for item in items:
                cur_results.append(item)
            print(f"{len(cur_results)} items fetched")
        with open(f"{OUTPUT_DIR}/{query.replace(' ', '_')}.json", "w") as f:
            json.dump(cur_results, f, indent=4)
        print(f"Fetched {len(cur_results)} items for query: {query}")
    


if __name__=='__main__':
    get_code_search_results()
    

    
