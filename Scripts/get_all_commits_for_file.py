import requests
import json
import csv
import time

# Load GitHub token
with open("config.json", "r") as f:
    cf = json.load(f)
    token = cf['gh_tokens'][0]  # Use the first token for simplicity

# GitHub API headers
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

def get_commits_for_file(repo,filepath):
    url = f"https://api.github.com/repos/{repo}/commits"
    params = {
        "path": filepath,
        "per_page": 100
    }

    all_commits = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, headers=headers, params=params)
        time.sleep(0.5)  # To respect rate limits
        if response.status_code != 200:
            print(f"Error for {filepath}: {response.status_code} - {response.text}")
            break

        data = response.json()
        if not data:
            break

        all_commits.extend(data)
        page += 1

    return all_commits

def process_tsv(tsv_file):
    with open(tsv_file, newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        iterr = 0
        rows = []
        for row in reader:
            iterr += 1
            
            if iterr == 1:  # Skip header row
                row.append("commits")  # Add a new column for commits
                rows.append(row)
                continue
            if not row:
                continue
            if iterr < 289:
                continue
            filepath = row[1]
            htmlpath = row[4]
            arr = htmlpath.split('/')
            print(htmlpath)
            print(arr)
            repo_full_path = arr[3] + '/' + arr[4]
            print(f"\nFetching commits for: {filepath}")
            commits = get_commits_for_file(repo_full_path, filepath)
            time.sleep(1)  # To respect rate limits
            cmms = []
            for commit in commits:
                sha = commit["sha"]
                author = commit["commit"]["author"]["name"]
                date = commit["commit"]["author"]["date"]
                message = commit["commit"]["message"]
                print(f"{filepath}\t{sha[:7]}\t{author}\t{date}\t{message}")
                cmms.append({
                    "sha": sha,
                    "author": author,
                    "date": date,
                    "message": message
                })
            row.append(json.dumps(cmms))  # Append commits to the row
            rows.append(row)
            print(f"Commits for {filepath} added to row.")
            if iterr % 10 == 0:
                print(f"Processed {iterr} rows, saving progress...")
                #save rows to tsv file along with header
                with open("DS/annotations/final_500/final_500_with_commits.tsv", 'w', newline='') as file:
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerows(rows)



                
        with open("DS/annotations/final_500/final_500_with_commits.tsv", 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            # file.seek(0)
            writer.writerows(rows)  # Write the updated rows back to the file
        print(f"Updated {tsv_file} with commit information.")
        #also write the rows to a json file
        # with open("DS/annotations/final_500/final_500_with_commits.json", 'w') as json_file:
        #     json.dump(reader, json_file, indent=4)  
    

# Example usage
if __name__ == "__main__":
    tsv_file = "DS/annotations/final_500/final_500.tsv"
    process_tsv(tsv_file)
