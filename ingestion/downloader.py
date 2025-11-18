import os
import requests
import zipfile
import io
from datetime import datetime

def repo_to_zip_url(repo_url, branch="master"):
    if repo_url.endswith("/"):
        repo_url = repo_url[:-1]
    return repo_url + f"/archive/refs/heads/{branch}.zip"

def download_and_extract(repo_url, base_dir="repos", branch="master"):
    os.makedirs(base_dir, exist_ok=True)
    repo_name=repo_url.rstrip("/").split("/")[-1]
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")

    target_dir = os.path.join(base_dir, f"{repo_name}{timestamp}")
    zip_url = repo_to_zip_url(repo_url, branch)
    print(f"[INFO]: downloading from: {zip_url}")
    response = requests.get(zip_url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(target_dir)
        entries = os.listdir(target_dir)
        if len(entries) == 1:
            single=os.path.join(target_dir, entries[0])
            if os.path.isdir(single):
                return single

    print(f"[SUCCESS]: extracted to {target_dir}")
    return target_dir
    

