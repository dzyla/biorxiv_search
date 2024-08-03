import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import requests
import json
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import torch
import shutil
import dropbox
import streamlit as st
import time

def retry_on_exception(exception, retries=5, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(retries):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    last_exception = e
                    print(f"Retrying due to: {str(e)}")
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry_on_exception(requests.exceptions.ConnectionError)
def fetch_and_save_data_block(endpoint, server, block_start, block_end, save_directory, format='json'):
    base_url = f"https://api.medrxiv.org/details/{server}/"
    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True

    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/{format}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch data for block {block_interval} at cursor {cursor}. HTTP Status: {response.status_code}")
            break

        data = response.json()
        fetched_papers = len(data['collection'])

        if fetched_papers > 0:
            block_data.extend(data['collection'])
            cursor += fetched_papers
            print(f"Fetched {fetched_papers} papers for block {block_interval}. Total fetched: {cursor}.")
        else:
            continue_fetching = False

    if block_data:
        save_data_block(block_data, block_start, block_end, endpoint, save_directory)

def save_data_block(block_data, start_date, end_date, endpoint, save_directory):
    start_yymmdd = start_date.strftime("%y%m%d")
    end_yymmdd = end_date.strftime("%y%m%d")
    filename = f"{save_directory}/{endpoint}_data_{start_yymmdd}_{end_yymmdd}.json"
    
    with open(filename, 'w') as file:
        json.dump(block_data, file, indent=4)
    
    print(f"Saved data block to {filename}")

def fetch_data(endpoint, server, interval, save_directory, format='json'):
    os.makedirs(save_directory, exist_ok=True)
    start_date, end_date = [datetime.strptime(date, "%Y-%m-%d") for date in interval.split('/')]
    current_date = start_date
    tasks = []

    with ThreadPoolExecutor(max_workers=12) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)
            tasks.append(executor.submit(fetch_and_save_data_block, endpoint, server, block_start, block_end, save_directory, format))
            current_date += relativedelta(months=1)
        
        for future in as_completed(tasks):
            future.result()

def load_json_to_dataframe(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def save_dataframe(df, save_path):
    df.to_parquet(save_path)

def process_json_files(directory, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    
    json_files = list(Path(directory).glob('*.json'))
    print(f'json_files {type(json_files)}: {json_files}')
    
    for json_file in json_files:
        df = load_json_to_dataframe(json_file)
        
        parquet_filename = f"{json_file.stem}.parquet"
        save_path = os.path.join(save_directory, parquet_filename)
        
        if os.path.exists(save_path):
            npy_file_path = save_path.replace('db_update', 'embed_update').replace('parquet', 'npy')
            if os.path.exists(npy_file_path):
                os.remove(npy_file_path)
                print(f'Removed embedding file {npy_file_path} due to the dataframe update')
                
        save_dataframe(df, save_path)
        print(f"Processed and saved {json_file.name} to {parquet_filename}")

def load_unprocessed_parquets(db_update_directory, embed_update_directory):
    db_update_directory = Path(db_update_directory)
    embed_update_directory = Path(embed_update_directory)

    parquet_files = list(db_update_directory.glob('*.parquet'))
    npy_files = {f.stem for f in embed_update_directory.glob('*.npy')}
    unprocessed_dataframes = []

    for parquet_file in parquet_files:
        if parquet_file.stem not in npy_files:
            unprocessed_dataframes.append(parquet_file)
            print(f"Loaded unprocessed Parquet file: {parquet_file.name}")
        else:
            print(f"Skipping processed Parquet file: {parquet_file.name}")

    return unprocessed_dataframes

def connect_to_dropbox():
    dropbox_APP_KEY = st.secrets["dropbox_APP_KEY"]
    dropbox_APP_SECRET = st.secrets["dropbox_APP_SECRET"]
    dropbox_REFRESH_TOKEN = st.secrets["dropbox_REFRESH_TOKEN"]
    
    dbx = dropbox.Dropbox(
        app_key=dropbox_APP_KEY,
        app_secret=dropbox_APP_SECRET,
        oauth2_refresh_token=dropbox_REFRESH_TOKEN
    )
    return dbx

def upload_path(local_path, dropbox_path):
    dbx = connect_to_dropbox()
    local_path = Path(local_path)

    if local_path.is_file():
        relative_path = local_path.name
        dropbox_file_path = os.path.join(dropbox_path, relative_path).replace('\\', '/').replace('//', '/')
        upload_file(local_path, dropbox_file_path, dbx)
    elif local_path.is_dir():
        for local_file in local_path.rglob('*'):
            if local_file.is_file():
                relative_path = local_file.relative_to(local_path.parent)
                dropbox_file_path = os.path.join(dropbox_path, relative_path).replace('\\', '/').replace('//', '/')
                upload_file(local_file, dropbox_file_path, dbx)
    else:
        print("The provided path does not exist.")

def upload_file(file_path, dropbox_file_path, dbx):
    try:
        dropbox_file_path = dropbox_file_path.replace('\\', '/')

        try:
            metadata = dbx.files_get_metadata(dropbox_file_path)
            dropbox_mod_time = metadata.server_modified
            local_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)

            if dropbox_mod_time >= local_mod_time:
                print(f"Skipped {dropbox_file_path}, Dropbox version is up-to-date.")
                return
        except dropbox.exceptions.ApiError as e:
            if not isinstance(e.error, dropbox.files.GetMetadataError) or e.error.is_path() and e.error.get_path().is_not_found():
                print(f"No existing file on Dropbox, proceeding with upload: {dropbox_file_path}")
            else:
                raise e

        with file_path.open('rb') as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"Uploaded {dropbox_file_path}")
    except Exception as e:
        print(f"Failed to upload {dropbox_file_path}: {str(e)}")

def load_data_embeddings():
    new_data_directory = "db_update_med"
    updated_embeddings_directory = "embed_update_med"
    new_data_files = sorted(Path(new_data_directory).glob("*.parquet"))
    
    df_updates_list = []
    embeddings_updates_list = []
    
    for data_file in new_data_files:
        # Assuming naming convention allows direct correlation
        corresponding_embedding_file = Path(updated_embeddings_directory) / (
            data_file.stem + ".npy"
        )

        if corresponding_embedding_file.exists():
            # Load and append DataFrame and embeddings
            df_updates_list.append(pd.read_parquet(data_file))
            embeddings_updates_list.append(np.load(corresponding_embedding_file))
        else:
            print(f"No corresponding embedding file found for {data_file.name}")

    new_data_files = sorted(Path(new_data_directory).glob("*.parquet"))
    for data_file in new_data_files:
        corresponding_embedding_file = Path(updated_embeddings_directory) / (
            data_file.stem + ".npy"
        )

        if corresponding_embedding_file.exists():
            df_updates_list.append(pd.read_parquet(data_file))
            embeddings_updates_list.append(np.load(corresponding_embedding_file))
        else:
            print(f"No corresponding embedding file found for {data_file.name}")

    if df_updates_list:
        df_updates = pd.concat(df_updates_list)
    else:
        df_updates = pd.DataFrame()

    if embeddings_updates_list:
        embeddings_updates = np.vstack(embeddings_updates_list)
    else:
        embeddings_updates = np.array([])

    df_combined = df_updates
    mask = ~df_combined.duplicated(subset=["title"], keep="last")
    df_combined = df_combined[mask]

    embeddings_combined = embeddings_updates

    embeddings_combined = embeddings_combined[mask]

    return df_combined, embeddings_combined

endpoint = "details"
server = "medrxiv"

df, embeddings = load_data_embeddings()

try:
    start_date = df['date'].max()
except:
    start_date = '1990-01-01'
last_date = datetime.today().strftime('%Y-%m-%d')

interval = f'{start_date}/{last_date}'
print(f'using interval: {interval}')

save_directory = "db_update_json_med"
fetch_data(endpoint, server, interval, save_directory)

directory = r'db_update_json_med'
save_directory = r'db_update_med'
process_json_files(directory, save_directory)

db_update_directory = 'db_update_med'
embed_update_directory = 'embed_update_med'
unprocessed_dataframes = load_unprocessed_parquets(db_update_directory, embed_update_directory)

if unprocessed_dataframes:
    for file in unprocessed_dataframes:
        df = pd.read_parquet(file)
        query = df['abstract'].tolist()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        model.to(device)

        query_embedding = model.encode(query, normalize_embeddings=True, precision='ubinary', show_progress_bar=True)
        file_path = os.path.basename(file).split('.')[0]
        os.makedirs('embed_update_med', exist_ok=True)
        embeddings_path = f'embed_update_med/{file_path}'
        np.save(embeddings_path, query_embedding)
        print(f'Saved embeddings {embeddings_path}')

    db_update_json = 'db_update_json_med'
    shutil.rmtree(db_update_json)
    print(f"Directory '{db_update_json}' and its contents have been removed.")
    
    for path in ['db_update_med', 'embed_update_med']:
        upload_path(path, '/')

else:
    print('Nothing to do')
