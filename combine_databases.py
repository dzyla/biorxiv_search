import pandas as pd
import numpy as np
from pathlib import Path

def combine_databases():
    # Define paths
    aggregated_data_path = Path("aggregated_data")
    db_update_bio_path = Path("db_update")
    biorxiv_embeddings_path = Path("biorxiv_ubin_embaddings.npy")
    embed_update_bio_path = Path("embed_update")

    db_update_med_path = Path("db_update_med")
    embed_update_med_path = Path("embed_update_med")

    # Load existing database and embeddings for BioRxiv
    df_bio_existing = pd.read_parquet(aggregated_data_path)
    bio_embeddings_existing = np.load(biorxiv_embeddings_path, allow_pickle=True)
    print(f"Existing BioRxiv data shape: {df_bio_existing.shape}, Existing BioRxiv embeddings shape: {bio_embeddings_existing.shape}")

    # Determine the embedding size from existing embeddings
    embedding_size = bio_embeddings_existing.shape[1]

    # Prepare lists to collect new updates
    bio_dfs_list = []
    bio_embeddings_list = []

    # Helper function to process updates from a specified directory
    def process_updates(new_data_directory, updated_embeddings_directory, dfs_list, embeddings_list, server_name):
        new_data_files = sorted(Path(new_data_directory).glob("*.parquet"))
        for data_file in new_data_files:
            corresponding_embedding_file = Path(updated_embeddings_directory) / (data_file.stem + ".npy")

            if corresponding_embedding_file.exists():
                df = pd.read_parquet(data_file)
                df['server'] = server_name  # Add server column
                new_embeddings = np.load(corresponding_embedding_file, allow_pickle=True)

                # Check if the number of rows in the DataFrame matches the number of rows in the embeddings
                if df.shape[0] != new_embeddings.shape[0]:
                    print(f"Shape mismatch for {data_file.name}: DataFrame has {df.shape[0]} rows, embeddings have {new_embeddings.shape[0]} rows. Skipping.")
                    continue

                # Check embedding size and adjust if necessary
                if new_embeddings.shape[1] != embedding_size:
                    print(f"Skipping {data_file.name} due to embedding size mismatch.")
                    continue

                dfs_list.append(df)
                embeddings_list.append(new_embeddings)
            else:
                print(f"No corresponding embedding file found for {data_file.name}")

    # Process updates from both BioRxiv and MedRxiv
    process_updates(db_update_bio_path, embed_update_bio_path, bio_dfs_list, bio_embeddings_list, 'biorxiv')
    process_updates(db_update_med_path, embed_update_med_path, bio_dfs_list, bio_embeddings_list, 'medrxiv')

    # Concatenate all BioRxiv and MedRxiv updates
    if bio_dfs_list:
        df_combined = pd.concat(bio_dfs_list)
    else:
        df_combined = pd.DataFrame()

    if bio_embeddings_list:
        embeddings_combined = np.vstack(bio_embeddings_list)
    else:
        embeddings_combined = np.array([])

    # Append new data to existing, handling duplicates as needed
    df_combined = pd.concat([df_bio_existing, df_combined])

    # Create a mask for filtering unique titles
    mask = ~df_combined.duplicated(subset=["title"], keep="last")
    df_combined = df_combined[mask]

    # Combine embeddings, ensuring alignment with the DataFrame
    embeddings_combined = (
        np.vstack([bio_embeddings_existing, embeddings_combined])
        if embeddings_combined.size
        else bio_embeddings_existing
    )

    # Filter the embeddings based on the DataFrame unique entries
    embeddings_combined = embeddings_combined[mask]

    assert df_combined.shape[0] == embeddings_combined.shape[0], "Shape mismatch between DataFrame and embeddings"

    print(f"Filtered DataFrame shape: {df_combined.shape}")
    print(f"Filtered embeddings shape: {embeddings_combined.shape}")

    # Save combined DataFrame and embeddings
    combined_data_path = aggregated_data_path / "combined_data.parquet"
    df_combined.to_parquet(combined_data_path)
    print(f"Saved combined DataFrame to {combined_data_path}")

    combined_embeddings_path = "combined_ubin_embeddings.npy"
    np.save(combined_embeddings_path, embeddings_combined)
    print(f"Saved combined embeddings to {combined_embeddings_path}")

if __name__ == "__main__":
    combine_databases()
