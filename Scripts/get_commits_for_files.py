import os
import tempfile
import argparse
import json
from utils.gh_api_manager import GH_API_Manager
import time
import csv
from pathlib import Path
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


input_file = "DS/complete/ds_comments_complete.json"


def get_files():


    files = []
    seen = {}
    with open(input_file,"r") as f:
        data = json.load(f)
        for d in data:
            repo = d["repo_full_name"]
            path = d["path"]
            key = repo + "/" + path
            if key in seen:
                continue
            seen[key] = 1
            files.append(d)
    print(f"No of files:: {len(files)}")
    return files



def get_branches(repo_full_name):    
    """
    Get all branches of a repository.
    :param repo_url: URL of the repository
    :return: List of branch names
    """
    url = f"https://api.github.com/repos/{repo_full_name}/branches"
    response = gh_api_manager.make_request(url)
    if response and len(response) > 0:
        return [branch['name'] for branch in response]
    else:
        print(f"Error fetching branches: {repo_full_name}")
        return []


branch_not_found = 0

def get_commits(repo_full_name, file_path):
    page = 0
    params = {
        'path': file_path,
        'per_page': 100,
        'page': page
    }
    url = f"https://api.github.com/repos/{repo_full_name}/commits"
    commits = []
    branches = get_branches(repo_full_name)
    # branches.append("#")
    if not branches:
        branches = ['main']  # Fallback to main branch if no branches found
        print(f"No branches found for repository {repo_full_name}.")
        global branch_not_found
        branch_not_found += 1
        # return commits
    seen = {}
    for branch in branches:
        while True:
            # print(f"Working on branch : {branch}")
            # if branch == "main" or branch=="#":
            params['sha'] = branch
            response = gh_api_manager.make_request(url, params)
            if not response or len(response) == 0:
                break
            # print(len(response))
            for commit in response:
                shaa = commit['sha']
                if shaa in seen:
                    continue
                seen[shaa] = 1
                commit_url = f"https://api.github.com/repos/{repo_full_name}/commits/{shaa}"
                commit_details = gh_api_manager.make_request(commit_url)
                time.sleep(0.05)  # To respect rate limits
                print(f"requesting :: {commit_url}")
                if commit_details and 'files' in commit_details:
                    for file in commit_details['files']:
                        if file['filename'] == file_path:
                            pat = ""
                            if "patch" in file:
                                pat = file["patch"]
                            commits.append({
                                'sha': shaa,
                                'message': commit['commit']['message'],
                                'author': commit['commit']['author']['name'],
                                'date': commit['commit']['author']['date'],
                                'patch': pat,
                                'url': commit_url,
                                'branch': branch
                            })
            
            page += 1
            params['page'] = page
    return commits






def wrt(data):
    with open("DS/complete/commits_for_files2.json","w") as f:
        json.dump(data,f,indent=4)


def main():

    files = get_files()
    print(len(files))


    data = {}
    inc = 0
    
    # with open("DS/complete/commits_for_files2.json","r") as f:
    #     pdata = json.load(f)

    # plen = len(pdata) 
    plen = 0
    # data = pdata
    for d in files:
        inc += 1
        if(inc < plen):
            continue
        repo = d["repo_full_name"]
        path = d["path"]
        key = repo + "/" + path
        data[key] = get_commits(repo,path)
        print(f"For file: {path}, in repo: {repo}, found commits :: {len(data[key])}")
        if inc % 5 == 0:
            wrt(data)


    wrt(data)


main()
# def m2():