import streamlit as st
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer, models
import torch
from sentence_transformers.quantization import semantic_search_faiss
from pathlib import Path
import time
import plotly.express as px
import doi
import requests
from groq import Groq
import dropbox
from datetime import datetime, timedelta


API_URL = (
    "https://api-inference.huggingface.co/models/mixedbread-ai/mxbai-embed-large-v1"
)
summarization_API_URL = (
    "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
)

LLM_API_URL = (
    "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
)

API_TOKEN = st.secrets["hf_token"]  # Replace with your Hugging Face API token

headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query_hf_api(text, api=API_URL, parameters=None):
    
    if not parameters:
        payload = {"inputs": text}
    else:
        payload = {
            "inputs": text,
            "parameters": parameters,
        }

    response = requests.post(api, headers=headers, json=payload)

    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError:
        st.error("Failed to get a valid response from the server. Please try again later.")
        return {}

    # Prepare an empty placeholder that can be filled if needed
    progress_placeholder = st.empty()

    # Check if the model is currently loading
    if "error" in response_data and "loading" in response_data["error"]:
        estimated_time = response_data.get("estimated_time", 30)  # Default wait time to 30 seconds if not provided
        with progress_placeholder.container():
            st.warning(f"Model from :hugging_face: is currently loading. Estimated wait time: {estimated_time:.1f} seconds. Please wait...")
            
            # Create a progress bar within the container
            progress_bar = st.progress(0)
            for i in range(int(estimated_time) + 5):  # Adding a buffer time to ensure the model is loaded
                # Update progress bar. The factor of 100 is used to convert to percentage completion
                progress = int((i / (estimated_time + 5)) * 100)
                progress_bar.progress(progress)
                time.sleep(1)  # Wait for a second

        # Clear the placeholder once loading is complete
        progress_placeholder.empty()
        
        st.rerun()  # Rerun the app after waiting

    return response_data


def normalize_embeddings(embeddings):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
    embeddings (Tensor): The embeddings tensor to normalize.

    Returns:
    Tensor: The normalized embeddings.
    """
    if embeddings.dim() == 1:
        # Add an extra dimension if the tensor is 1-dimensional
        embeddings = embeddings.unsqueeze(0)
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def quantize_embeddings(
    embeddings, precision="ubinary", ranges=None, calibration_embeddings=None
):
    """
    Quantizes embeddings to a specified precision using PyTorch and numpy.

    Args:
    embeddings (Tensor): The embeddings to quantize, assumed to be a Tensor.
    precision (str): The precision to convert to.
    ranges (np.ndarray, optional): Ranges for quantization.
    calibration_embeddings (Tensor, optional): Embeddings used for calibration.

    Returns:
    Tensor: The quantized embeddings.
    """
    if precision == "float32":
        return embeddings.float()

    if precision in ["int8", "uint8"]:
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = torch.stack(
                    (
                        torch.min(calibration_embeddings, dim=0)[0],
                        torch.max(calibration_embeddings, dim=0)[0],
                    )
                )
            else:
                ranges = torch.stack(
                    (torch.min(embeddings, dim=0)[0], torch.max(embeddings, dim=0)[0])
                )

        starts, ends = ranges[0], ranges[1]
        steps = (ends - starts) / 255

        if precision == "uint8":
            quantized_embeddings = torch.clip(
                ((embeddings - starts) / steps), 0, 255
            ).byte()
        elif precision == "int8":
            quantized_embeddings = torch.clip(
                ((embeddings - starts) / steps - 128), -128, 127
            ).char()

    elif precision == "binary" or precision == "ubinary":
        embeddings_np = embeddings.numpy() > 0
        packed_bits = np.packbits(embeddings_np, axis=-1)
        if precision == "binary":
            quantized_embeddings = torch.from_numpy(packed_bits - 128).char()
        else:
            quantized_embeddings = torch.from_numpy(packed_bits).byte()

    else:
        raise ValueError(f"Precision {precision} is not supported")

    return quantized_embeddings


def process_embeddings(embeddings, precision="ubinary", calibration_embeddings=None):
    """
    Normalizes and quantizes embeddings from an API list to a specified precision using PyTorch.

    Args:
    embeddings (list or Tensor): Raw embeddings from an external API, either as a list or a Tensor.
    precision (str): Desired precision for quantization.
    calibration_embeddings (Tensor, optional): Embeddings for calibration.

    Returns:
    Tensor: Processed embeddings, normalized and quantized.
    """
    # Convert list to Tensor if necessary
    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    elif not isinstance(embeddings, torch.Tensor):
        st.error(embeddings)
        raise TypeError(
            f"Embeddings must be a list or a torch.Tensor. Message from the server: {embeddings}"
        )

    # Convert calibration_embeddings list to Tensor if necessary
    if isinstance(calibration_embeddings, list):
        calibration_embeddings = torch.tensor(
            calibration_embeddings, dtype=torch.float32
        )
    elif calibration_embeddings is not None and not isinstance(
        calibration_embeddings, torch.Tensor
    ):
        raise TypeError(
            "Calibration embeddings must be a list or a torch.Tensor if provided. "
        )

    normalized_embeddings = normalize_embeddings(embeddings)
    quantized_embeddings = quantize_embeddings(
        normalized_embeddings,
        precision=precision,
        calibration_embeddings=calibration_embeddings,
    )
    return quantized_embeddings.cpu().numpy()

def connect_to_dropbox():
    dropbox_APP_KEY = st.secrets["dropbox_APP_KEY"]
    dropbox_APP_SECRET = st.secrets["dropbox_APP_SECRET"]
    dropbox_REFRESH_TOKEN = st.secrets["dropbox_REFRESH_TOKEN"]
    
    dbx = dbx = dropbox.Dropbox(
            app_key = dropbox_APP_KEY,
            app_secret = dropbox_APP_SECRET,
            oauth2_refresh_token = dropbox_REFRESH_TOKEN
        )
    return dbx

def list_files(dropbox_path):
    dbx = connect_to_dropbox()
    files = []
    try:
        response = dbx.files_list_folder(dropbox_path)
        files = response.entries
    except Exception as e:
        st.error(f"Failed to list files: {str(e)}")
    return files

def download_folder(dropbox_path, local_path):
    placeholder = st.empty()
    dbx = connect_to_dropbox()
    try:
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        response = dbx.files_list_folder(dropbox_path)
        total_files = len(response.entries)
        if total_files == 0:
            return

        current_file = 0
        for entry in response.entries:
            local_file_path = Path(local_path) / entry.name
            if isinstance(entry, dropbox.files.FileMetadata):
                # Only download if the file does not exist locally
                if not local_file_path.exists():
                    placeholder.write(f'Downloading {entry.name}')
                    dbx.files_download_to_file(str(local_file_path), entry.path_lower)
            elif isinstance(entry, dropbox.files.FolderMetadata):
                # Recursively download contents of the directory
                download_folder(entry.path_lower, str(local_file_path))

            current_file += 1
        placeholder.empty()
    except Exception as e:
        st.error(f"Failed to download: {str(e)}")
    placeholder.empty()

def download_data_from_dropbox():
    # Check if 'last_download_time' is in the session state and if 24 hours have passed
    if True:
        placeholder = st.empty()
        placeholder.write('Downloading data...')
        
        local_path = os.getcwd()
        
        # Run the download function
        download_folder('//', local_path)
        
        # Update the session state with the current time
        st.session_state.last_download_time = datetime.now()
        
        placeholder.write("Download completed and data updated.")
        placeholder.empty()

# Load data and embeddings
@st.cache_resource(ttl="1d")
def load_data_embeddings():
    existing_data_path = "aggregated_data"
    new_data_directory_bio = "db_update"
    existing_embeddings_path = "biorxiv_ubin_embaddings.npy"
    updated_embeddings_directory_bio = "embed_update"

    new_data_directory_med = "db_update_med"
    updated_embeddings_directory_med = "embed_update_med"
    
    # Load existing database and embeddings
    df_existing = pd.read_parquet(existing_data_path)
    embeddings_existing = np.load(existing_embeddings_path, allow_pickle=True)

    print(f"Existing data shape: {df_existing.shape}, Existing embeddings shape: {embeddings_existing.shape}")

    # Determine the embedding size from existing embeddings
    embedding_size = embeddings_existing.shape[1]

    # Prepare lists to collect new updates
    df_updates_list = []
    embeddings_updates_list = []

    # Helper function to process updates from a specified directory
    def process_updates(new_data_directory, updated_embeddings_directory):
        new_data_files = sorted(Path(new_data_directory).glob("*.parquet"))
        print(new_data_files)
        for data_file in new_data_files:
            corresponding_embedding_file = Path(updated_embeddings_directory) / (
                data_file.stem + ".npy"
            )

            if corresponding_embedding_file.exists():
                df = pd.read_parquet(data_file)
                new_embeddings = np.load(corresponding_embedding_file, allow_pickle=True)
                
                # Check if the number of rows in the DataFrame matches the number of rows in the embeddings
                if df.shape[0] != new_embeddings.shape[0]:
                    print(f"Shape mismatch for {data_file.name}: DataFrame has {df.shape[0]} rows, embeddings have {new_embeddings.shape[0]} rows. Skipping.")
                    continue
                
                # Check embedding size and adjust if necessary
                if new_embeddings.shape[1] != embedding_size:
                    print(f"Skipping {data_file.name} due to embedding size mismatch.")
                    continue

                df_updates_list.append(df)
                embeddings_updates_list.append(new_embeddings)
            else:
                print(f"No corresponding embedding file found for {data_file.name}")

    # Process updates from both BioRxiv and MedArXiv
    process_updates(new_data_directory_bio, updated_embeddings_directory_bio)
    process_updates(new_data_directory_med, updated_embeddings_directory_med)

    # Concatenate all updates
    if df_updates_list:
        df_updates = pd.concat(df_updates_list)
    else:
        df_updates = pd.DataFrame()

    if embeddings_updates_list:
        embeddings_updates = np.vstack(embeddings_updates_list)
    else:
        embeddings_updates = np.array([])

    # Append new data to existing, handling duplicates as needed
    df_combined = pd.concat([df_existing, df_updates])

    # Create a mask for filtering
    mask = ~df_combined.duplicated(subset=["title"], keep="last")
    df_combined = df_combined[mask]

    # Combine embeddings, ensuring alignment with the DataFrame
    embeddings_combined = (
        np.vstack([embeddings_existing, embeddings_updates])
        if embeddings_updates.size
        else embeddings_existing
    )

    # Filter the embeddings based on the dataframe unique entries
    embeddings_combined = embeddings_combined[mask]

    return df_combined, embeddings_combined


LLM_prompt = "Review the abstracts listed below and create a list and summary that captures their main themes and findings. Identify any commonalities across the abstracts and highlight these in your summary. Ensure your response is concise, avoids external links, and is formatted in markdown.\n\n"

def summarize_abstract(abstract, llm_model="llama-3.1-70b-versatile", instructions=LLM_prompt, api_key=st.secrets["groq_token"]):
    """
    Summarizes the provided abstract using a specified LLM model.
    
    Parameters:
    - abstract (str): The abstract text to be summarized.
    - llm_model (str): The LLM model used for summarization. Defaults to "llama-3.1-70b-versatile".
    
    Returns:
    - str: A summary of the abstract, condensed into one to two sentences.
    """
    # Initialize the Groq client with the API key from environment variables
    client = Groq(api_key=api_key)
    
    formatted_text = "\n".join(f"{idx + 1}. {abstract}" for idx, abstract in enumerate(abstracts))
    try:
        # Create a chat completion with the abstract and specified LLM model
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": f'{instructions} "{formatted_text}"'}],
            model=llm_model,
        )
    except:
        return 'Groq model not available or above the usage limit. Use own API key from here: https://console.groq.com/keys'
        
    
    # Return the summarized content
    return chat_completion.choices[0].message.content

### To use with local setup
# @st.cache_resource()
# def model_to_device():
#     # Determine the device to use: use CUDA if available; otherwise, use CPU.
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
#     model.to(device)

#     return model


def define_style():
    st.markdown(
        """
    <style>
        .stExpander > .stButton > button {
            width: 100%;
            border: none;
            background-color: #f0f2f6;
            color: #333;
            text-align: left;
            padding: 15px;
            font-size: 18px;
            border-radius: 10px;
            margin-top: 5px;
        }
        .stExpander > .stExpanderContent {
            padding-left: 10px;
            padding-top: 10px;
        }
        a {
            color: #FF4B4B;
            text-decoration: none;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def logo(db_update_date, db_size_bio, db_size_med):
    # Initialize Streamlit app
    biorxiv_logo = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; align-items: center;'>
            <div style='margin-right: 20px;'>
                <img src='{biorxiv_logo}' alt='BioRxiv logo' style='max-height: 100px;'>
            </div>
            <div style='margin-left: 20px;'>
                <img src='{medarxiv_logo}' alt='medRxiv logo' style='max-height: 100px;'>
            </div>
        </div>
        <div style='text-align: center; margin-top: 10px;'>
            <h3 style='color: black;'>Manuscript Semantic Search [bMSS]</h3>
            Last database update: {db_update_date}; Database size: bioRxiv: {db_size_bio} / medRxiv: {db_size_med} entries
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="bMSS",
    page_icon=":scroll:",
)
download_data_from_dropbox()
define_style()

df, embeddings_unique = load_data_embeddings()
logo(df["date"].max(), df[df['server']=='biorxiv'].shape[0], df[df['server']=='medrxiv'].shape[0])

# model = model_to_device()

corpus_index = None
corpus_precision = "ubinary"
use_hf = False 

query = st.text_input("Enter your search query:")

col1, col2 = st.columns(2)
with col1:
    num_to_show = st.number_input(
        "Number of results to show:", min_value=1, max_value=50, value=10
    )
with col2:
    use_ai = st.checkbox('Use AI generated summary?')

if use_ai:
    with col2:
        groq_api_provided = st.text_input('Own Groq API KEY to remove limits', '', help='To obtain own Groq key go to https://console.groq.com/keys')
        if not groq_api_provided:
            groq_api_provided = st.secrets["groq_token"]
        #use_hf = st.checkbox('Use free HF gemma 2B instead? (poor quality)')


if query:
    with st.spinner("Searching..."):
        # Encode the query
        search_start_time = time.time()
        # query_embedding = model.encode([query], normalize_embeddings=True, precision=corpus_precision)
        embedding_time = time.time()

        raw_embadding = query_hf_api(query)
        query_embedding = process_embeddings(raw_embadding)

        embedding_time_total = time.time() - embedding_time

        # Perform the search
        results, search_time, corpus_index = semantic_search_faiss(
            query_embedding,
            corpus_index=corpus_index,
            corpus_embeddings=embeddings_unique if corpus_index is None else None,
            corpus_precision=corpus_precision,
            top_k=num_to_show,  # type: ignore
            calibration_embeddings=None,
            rescore=False,
            rescore_multiplier=4,
            exact=True,
            output_index=True,
        )

        search_end_time = time.time()
        search_duration = search_end_time - search_start_time

        st.markdown(
            f"<h6 style='text-align: center; color: #7882af;'>Search Completed in {search_duration:.2f} seconds (embeddings time: {embedding_time_total:.2f})</h3>",
            unsafe_allow_html=True,
        )

        # Prepare the results for plotting
        plot_data = {"Date": [], "Title": [], "Score": [], "DOI": [], "category": [], "server": []}

        search_df = pd.DataFrame(results[0])

        # Find the minimum and maximum original scores
        min_score = search_df["score"].min()
        max_score = search_df["score"].max()

        # Normalize scores. The best score (min_score) becomes 100%, and the worst score (max_score) gets a value above 0%.
        search_df["score"] = abs(search_df["score"] - max_score) + min_score

        abstracts = []
        
        # Iterate over each row in the search_df DataFrame
        for index, entry in search_df.iterrows():
            row = df.iloc[int(entry["corpus_id"])]

            # Construct the DOI link
            try:
                doi_link = f"{doi.get_real_url_from_doi(row['doi'])}"
            except:
                doi_link = f'https://www.doi.org/'+row['doi']

            # Append information to plot_data for visualization
            plot_data["Date"].append(row["date"])
            plot_data["Title"].append(row["title"])
            plot_data["Score"].append(search_df["score"][index])  # type: ignore
            plot_data["DOI"].append(row["doi"])
            plot_data["category"].append(row["category"])
            plot_data["server"].append(row["server"])

            #summary_text = summarize_abstract(row['abstract'])

            with st.expander(f"{index+1}\. {row['title']}"): # type: ignore
                col1, col2 = st.columns(2)
                col1.markdown(f"**Score:** {entry['score']:.1f}")
                col2.markdown(f"**Server:** [{row['server']}]")
                st.markdown(f"**Authors:** {row['authors']}")
                col1, col2 = st.columns(2)
                col2.markdown(f"**Category:** {row['category']}")
                col1.markdown(f"**Date:** {row['date']}")
                #st.markdown(f"**Summary:**\n{summary_text}", unsafe_allow_html=False)
                abstracts.append(row['abstract'])
                st.markdown(
                    f"**Abstract:**\n{row['abstract']}", unsafe_allow_html=False
                )
                st.markdown(
                    f"**[Full Text Read]({doi_link})** 🔗", unsafe_allow_html=True
                )

        plot_df = pd.DataFrame(plot_data)

        # Convert 'Date' to datetime if it's not already in that format
        plot_df["Date"] = pd.to_datetime(plot_df["Date"])

        # Sort the DataFrame based on the Date to make sure it's ordered
        plot_df = plot_df.sort_values(by="Date")

        if use_ai:
            if not use_hf:
                ai_gen_start = time.time()
                st.markdown('**AI Summary of 10 abstracts:**')
                st.markdown(summarize_abstract(abstracts[:9], api_key=str(groq_api_provided)))
                total_ai_time = time.time()-ai_gen_start
                st.markdown(f'**Time to generate summary:** {total_ai_time:.2f} s')
            
            # Need to figure our how to get it from huggingface
            else:
                ai_gen_start = time.time()
                st.markdown('**AI Summary of 10 abstracts:**')
                formatted_text = str(LLM_prompt+"\n".join(f"{idx + 1}. {abstract}" for idx, abstract in enumerate(abstracts[:9])))
                prompt = f"Human: \n {formatted_text}\n\n AI:"
                LLM_answer = query_hf_api(formatted_text, summarization_API_URL)[0] #['generated_text']
                if 'AI:' in LLM_answer:
                    LLM_answer = LLM_answer.split('AI: ')[1]
                st.markdown(LLM_answer)
                total_ai_time = time.time()-ai_gen_start
                st.markdown(f'**Time to generate summary:** {total_ai_time:.2f} s')
            
        # Create a Plotly figure
        fig = px.scatter(
            plot_df,
            x="Date",
            y="Score",
            hover_data=["Title", "DOI"],
            color='server',
            title="Publication Times and Scores",
        )
        fig.update_traces(marker=dict(size=10))
        # Customize hover text to display the title and link it to the DOI
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b>",
            hovertext=plot_df.apply(lambda row: f"{row['Title']}", axis=1),
        )

        # Show the figure in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

        # Generate category counts for the pie chart
        category_counts = plot_df["category"].value_counts().reset_index()
        category_counts.columns = ["category", "count"]

        # Create a pie chart with Plotly Express
        fig = px.pie(
            category_counts,
            values="count",
            names="category",
            title="Category Distribution",
        )

        # Show the pie chart in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
        <div style='text-align: center;'>
            <b>Developed by <a href="https://www.dzyla.com/" target="_blank">Dawid Zyla</a></b>
            <br>
            <a href="https://github.com/dzyla/biorxiv_search" target="_blank">Source code on GitHub</a>
        </div>
        """,
    unsafe_allow_html=True,
)
