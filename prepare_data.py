import argparse
import pandas as pd
import numpy as np
import os
from collections import defaultdict


def prepare_kuaishou_data(interaction_path, title_path, video_path, output_dir):
    """
    Prepare Kuaishou dataset for DMMD4SR (Text + Visual modalities only)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(interaction_path):
        raise FileNotFoundError(f"Interaction file not found: {interaction_path}")
    if title_path and not os.path.exists(title_path):
        print(f"Warning: Title file not found: {title_path}")
        title_path = None
    if video_path and not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        video_path = None
    
    # Load interaction data
    print(f"Loading interaction data from: {interaction_path}")
    df = pd.read_parquet(interaction_path)
    print(f"Loaded {len(df)} interactions")
    
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # Create user sequences
    print("Creating user sequences...")
    user_seq = defaultdict(list)
    for _, row in df.iterrows():
        user_seq[int(row['user_id'])].append(int(row['item_id']))
    
    # Filter users with at least 3 interactions
    user_seq = {u: items for u, items in user_seq.items() if len(items) >= 3}
    
    # Save as txt format
    output_txt = os.path.join(output_dir, 'kuaishou.txt')
    print(f"Saving to {output_txt}...")
    
    with open(output_txt, 'w') as f:
        for user_id, items in sorted(user_seq.items()):
            f.write(f"{user_id} " + " ".join(map(str, items)) + "\n")
    
    print(f"✓ Saved {len(user_seq)} users")
    
    # Get max item_id
    max_item = max(max(items) for items in user_seq.values())
    print(f"Max item ID: {max_item}")
    
    # Load and save TEXT embeddings (title)
    if title_path:
        print("Processing TEXT embeddings (title)...")
        import torch
        
        title_emb = np.load(title_path)
        print(f"Loaded title embeddings: {title_emb.shape}")
        
        emb_dim = title_emb.shape[1]
        text_features = torch.zeros(max_item + 2, emb_dim)
        
        for i in range(min(len(title_emb), max_item + 1)):
            text_features[i + 1] = torch.from_numpy(title_emb[i].astype(np.float32))
        
        output_text = os.path.join(output_dir, 'text_features.pt')
        torch.save(text_features, output_text)
        print(f"✓ Saved text embeddings: {text_features.shape}")
    else:
        print("⚠ Skipping text embeddings (file not found)")
    
    # Load and save VISUAL embeddings (video_feature)
    if video_path:
        print("Processing VISUAL embeddings (video)...")
        import torch
        
        item_df = pd.read_parquet(video_path)
        print(f"Loaded item data: {len(item_df)} items")
        
        # Check if video_feature column exists
        if 'video_feature' not in item_df.columns:
            print(f"Warning: 'video_feature' column not found. Available columns: {item_df.columns.tolist()}")
            print("⚠ Skipping visual embeddings")
        else:
            sample_video = item_df['video_feature'].iloc[0]
            video_dim = len(sample_video)
            
            image_features = torch.zeros(max_item + 2, video_dim)
            
            # Map item_id to video_feature
            for _, row in item_df.iterrows():
                item_id = int(row['item_id']) + 1  # 1-based indexing
                if item_id <= max_item + 1:
                    video_feat = np.array(row['video_feature'], dtype=np.float32)
                    image_features[item_id] = torch.from_numpy(video_feat)
            
            output_image = os.path.join(output_dir, 'image_features.pt')
            torch.save(image_features, output_image)
            print(f"✓ Saved visual embeddings: {image_features.shape}")
    else:
        print("⚠ Skipping visual embeddings (file not found)")
    
    print("\n" + "="*60)
    print("Data preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"  - kuaishou.txt ({len(user_seq)} users)")
    if title_path:
        print(f"  - text_features.pt (TEXT: title)")
    if video_path and 'video_feature' in item_df.columns:
        print(f"  - image_features.pt (VISUAL: video)")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interaction', type=str, required=True,
                        help='Path to interaction.parquet')
    parser.add_argument('--title', type=str, default=None,
                        help='Path to title_emb.npy (TEXT modality)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to item_used.parquet (VISUAL modality)')
    parser.add_argument('--output', type=str, default='./dataset',
                        help='Output directory')
    
    args = parser.parse_args()
    
    prepare_kuaishou_data(
        interaction_path=args.interaction,
        title_path=args.title,
        video_path=args.video,
        output_dir=args.output
    )
