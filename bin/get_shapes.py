import h5py
from tqdm import tqdm
import numpy as np
import os
import concurrent.futures

feat_path = '/mnt/bulk/dferber/multimodal_llama/llama3/LLaVA-pp/LLaVA/llava/model/multimodal_encoder/FASTA_EMBEDDINGS/esm2_t36_3B_UR50D'

fs = [os.path.join(feat_path, f) for f in os.listdir(feat_path)]

lengths = []

def get_length(file_path):
    with h5py.File(file_path, 'r') as f:
        return f["features"].shape[0]

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(get_length, f): f for f in fs}
    with tqdm(total=len(fs), desc="Processing files") as pbar:
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")
            else:
                lengths.append(result)

print(f"Maximum: {np.max(lengths)}")
print(f"Minimum: {np.min(lengths)}")
print(f"Mean: {np.mean(lengths)}")
print(f"Median: {np.median(lengths)}")