import os
import argparse
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd

from proteinclip.swissprot import load_function_descriptions
from proteinclip.model_utils import ONNXModel

def build_parser():
    
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a pre-trained CLIP model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # /mnt/bulk/timlenz/tumpe/data/test_file_prot_clip.json

    parser.add_argument(
        "--feat_path",
        type=str,
        help="Path to feature files to extract embeddings from with pretrained protein projection network. Should be a folder contining metrics.csv and checkpoints/ folder.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        help="Path to CLIP model to extract projections. Should be a folder contining metrics.csv and checkpoints/ folder.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="Path to directory to save extracted projections. Defaults to subfolder 'projections' under CLIP dir.",
    )
    return parser

def main():
    args = build_parser().parse_args()
    
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    
    # protein embedding projector
    proj1 = ONNXModel(os.path.join(args.clip,"project_1.onnx"))
    # text projector
    #proj2 = ONNXModel(os.path.join(args.clip,"project_2.onnx"))
    
    #txt_embeds = load_function_descriptions(args.text_path)
    
    #df = pd.read_json(args.feat_json)
    
    #files = df.id.values
    
    for f in tqdm(os.listdir(args.feat_path)):
        assert os.path.exists(os.path.join(args.feat_path,f)), f'{os.path.join(args.feat_path,f)} Not found!'
        prot_emb = np.array(h5py.File(os.path.join(args.feat_path,f),'r')["embed_mean"][:])
        feats = proj1.predict(prot_emb)
        #txt_feats = proj2.predict(txt_embeds[f])
        with h5py.File(os.path.join(args.output_dir,f),'w') as h5f:
            h5f["prot_proj"] = feats
            #h5f["txt_proj"] = txt_feats
        
    
if __name__ == "__main__":
    main()
    
    