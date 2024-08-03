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
    def process_updates(new_data_directory, updated_embeddings_directory, dfs_list, embeddings_list):
        new_data_files = sorted(Path(new_data_directory).glob("*.parquet"))
        for data_file in new_data_files:
            corresponding_embedding_file = Path(updated_embeddings_directory) / (data_file.stem + ".npy")

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

                dfs_list.append(df)
                embeddings_list.append(new_embeddings)
            else:
                print(f"No corresponding embedding file found for {data_file.name}")

    # Process updates from both BioRxiv and MedRxiv
    process_updates(db_update_bio_path, embed_update_bio_path, bio_dfs_list, bio_embeddings_list)

    # Concatenate all BioRxiv updates
    if bio_dfs_list:
        df_bio_updates = pd.concat(bio_dfs_list)
    else:
        df_bio_updates = pd.DataFrame()

    if bio_embeddings_list:
        bio_embeddings_updates = np.vstack(bio_embeddings_list)
    else:
        bio_embeddings_updates = np.array([])

    # Append new BioRxiv data to existing, handling duplicates as needed
    df_bio_combined = pd.concat([df_bio_existing, df_bio_updates])

    # Create a mask for filtering unique titles
    bio_mask = ~df_bio_combined.duplicated(subset=["title"], keep="last")
    df_bio_combined = df_bio_combined[bio_mask]

    # Combine BioRxiv embeddings, ensuring alignment with the DataFrame
    bio_embeddings_combined = (
        np.vstack([bio_embeddings_existing, bio_embeddings_updates])
        if bio_embeddings_updates.size
        else bio_embeddings_existing
    )

    # Filter the embeddings based on the DataFrame unique entries
    bio_embeddings_combined = bio_embeddings_combined[bio_mask]

    assert df_bio_combined.shape[0] == bio_embeddings_combined.shape[0], "Shape mismatch between BioRxiv DataFrame and embeddings"

    print(f"Filtered BioRxiv DataFrame shape: {df_bio_combined.shape}")
    print(f"Filtered BioRxiv embeddings shape: {bio_embeddings_combined.shape}")

    # Save combined BioRxiv DataFrame and embeddings
    combined_biorxiv_data_path = aggregated_data_path / "combined_biorxiv_data.parquet"
    df_bio_combined.to_parquet(combined_biorxiv_data_path)
    print(f"Saved combined BioRxiv DataFrame to {combined_biorxiv_data_path}")

    combined_biorxiv_embeddings_path = "biorxiv_ubin_embaddings.npy"
    np.save(combined_biorxiv_embeddings_path, bio_embeddings_combined)
    print(f"Saved combined BioRxiv embeddings to {combined_biorxiv_embeddings_path}")

    # Prepare lists to collect new MedRxiv updates
    med_dfs_list = []
    med_embeddings_list = []

    process_updates(db_update_med_path, embed_update_med_path, med_dfs_list, med_embeddings_list)

    # Concatenate all MedRxiv updates
    if med_dfs_list:
        df_med_combined = pd.concat(med_dfs_list)
    else:
        df_med_combined = pd.DataFrame()

    if med_embeddings_list:
        med_embeddings_combined = np.vstack(med_embeddings_list)
    else:
        med_embeddings_combined = np.array([])

    last_date_in_med_database = df_med_combined['date'].max() if not df_med_combined.empty else "unknown"

    # Create a mask for filtering unique titles
    med_mask = ~df_med_combined.duplicated(subset=["title"], keep="last")
    df_med_combined = df_med_combined[med_mask]
    med_embeddings_combined = med_embeddings_combined[med_mask]

    assert df_med_combined.shape[0] == med_embeddings_combined.shape[0], "Shape mismatch between MedRxiv DataFrame and embeddings"

    print(f"Filtered MedRxiv DataFrame shape: {df_med_combined.shape}")
    print(f"Filtered MedRxiv embeddings shape: {med_embeddings_combined.shape}")

    # Save combined MedRxiv DataFrame and embeddings
    combined_medrxiv_data_path = db_update_med_path / f"database_{last_date_in_med_database}.parquet"
    df_med_combined.to_parquet(combined_medrxiv_data_path)
    print(f"Saved combined MedRxiv DataFrame to {combined_medrxiv_data_path}")

    combined_medrxiv_embeddings_path = embed_update_med_path / f"database_{last_date_in_med_database}.npy"
    np.save(combined_medrxiv_embeddings_path, med_embeddings_combined)
    print(f"Saved combined MedRxiv embeddings to {combined_medrxiv_embeddings_path}")

if __name__ == "__main__":
    combine_databases()
