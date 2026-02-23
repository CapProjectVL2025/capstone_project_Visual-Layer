#!/usr/bin/env python3
"""
ImageNet-1K Cleaning Script
===========================

Fetches ImageNet-1K from Visual Layer, applies cleaning policy,
and exports clean dataset for noise injection experiments.

Usage:
    # Set credentials
    export VL_API_KEY="your-api-key"
    export VL_API_SECRET="your-api-secret"
    
    # Run cleaning
    python clean_imagenet.py --dataset-id YOUR_DATASET_ID
    
    # Or with custom paths
    python clean_imagenet.py \
        --dataset-id YOUR_DATASET_ID \
        --policy cleaning_policy.yaml \
        --output ../data \
        --export ../clean_dataset_export

Requirements:
    pip install pandas pyarrow pyyaml requests PyJWT tqdm pillow
"""

import os
import sys
import argparse
import zipfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd
import yaml
from tqdm import tqdm

# Import Visual Layer API client
try:
    from clean_imagenet1k.connect_vl_api_f import VisualLayerAPIClient
except ModuleNotFoundError:
    from connect_vl_api_f import VisualLayerAPIClient

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Clean ImageNet-1K using Visual Layer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python clean_imagenet.py --dataset-id abc-123-def-456
    
    # Custom output paths
    python clean_imagenet.py \\
        --dataset-id abc-123-def-456 \\
        --policy my_policy.yaml \\
        --output ./data \\
        --export ./clean_export
    
    # Skip image download (metadata only)
    python clean_imagenet.py \\
        --dataset-id abc-123-def-456 \\
        --no-images
        """
    )
    
    parser.add_argument('--dataset-id', required=True,
                       help='Visual Layer dataset ID (find in URL when viewing dataset)')
    parser.add_argument('--policy', default='clean_imagenet1k/cleaning_policy.yaml',
                       help='Path to cleaning policy YAML (default: clean_imagenet1k/cleaning_policy.yaml)')
    parser.add_argument('--output', default='../data',
                       help='Output directory for cached data (default: ../data)')
    parser.add_argument('--export', default='../clean_dataset_export',
                       help='Export directory for clean dataset (default: ../clean_dataset_export)')
    parser.add_argument('--no-images', action='store_true',
                       help='Skip downloading images (metadata only)')
    parser.add_argument('--vm-path', default=None,
                       help='Copy final ZIP to VM path (e.g., /mnt/data/clean_imagenet)')
    parser.add_argument('--dry-run-auth', action='store_true',
                       help='Validate auth and dataset access only, then exit')
    
    return parser.parse_args()


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cleanup_python_cache(script_dir: Path):
    """Remove Python cache artifacts created during execution."""
    removed = 0
    for cache_dir in script_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed += 1

    for pyc_file in script_dir.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink(missing_ok=True)
            removed += 1

    if removed > 0:
        print(f"\n🧹 Cleaned Python cache artifacts: {removed}")


def load_policy(policy_path):
    """Load and validate cleaning policy."""
    print(f"📋 Loading policy: {policy_path}")
    
    if not Path(policy_path).exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    
    with open(policy_path) as f:
        policy = yaml.safe_load(f)
    
    print(f"  ✓ Policy loaded: {policy.get('policy_name', 'Unknown')}")
    print(f"    Uniqueness threshold: {policy.get('uniqueness_threshold', 'N/A')}")
    print(f"    Cluster dedup: {policy.get('dedupe_by_cluster', False)}")
    print(f"    Issue filters: {len(policy.get('drop_issues', []))}")
    print(f"    Tag filters: {len(policy.get('drop_tags', []))}")
    
    return policy


def export_from_visual_layer(client, dataset_id, output_dir, include_images):
    """Export dataset from Visual Layer."""
    print_header("Step 1: Export Dataset from Visual Layer")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start export
    export_task_id = client.export_dataset(
        dataset_id=dataset_id,
        format='json',
        include_images=include_images,
        file_name='imagenet_export.zip'
    )
    
    # Wait for completion
    download_url = client.wait_for_export(dataset_id, export_task_id)
    
    # Download
    export_zip = output_dir / "imagenet_export.zip"
    client.download_export(download_url, str(export_zip))
    
    # Extract
    print("📦 Extracting export...")
    extract_dir = output_dir / "vl_export"
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(export_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f"  ✓ Extracted to: {extract_dir}\n")
    
    return extract_dir


def parse_metadata(extract_dir):
    """Parse Visual Layer metadata."""
    print_header("Step 2: Parse Visual Layer Metadata")
    
    # Find metadata file
    metadata_files = list(extract_dir.glob("*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata file found in {extract_dir}")
    
    metadata_file = metadata_files[0]
    print(f"📖 Loading: {metadata_file.name}")
    
    # VL exports metadata as .json; handle both JSONL and JSON array/object.
    try:
        df = pd.read_json(metadata_file, lines=True)
        if 'metadata_items' not in df.columns and len(df) <= 1:
            raise ValueError("Likely non-JSONL JSON")
    except Exception:
        df = pd.read_json(metadata_file, lines=False)
    print(f"  ✓ Loaded {len(df):,} images")
    
    # Check for required columns
    if 'metadata_items' not in df.columns:
        raise ValueError("metadata_items column not found in export")
    
    # Parse metadata_items
    print("  ⏳ Parsing metadata_items...")
    
    def parse_items(items):
        if not isinstance(items, list):
            return [], [], []
        
        issues, tags, labels = [], [], []
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            item_type = item.get('type', '')
            
            if item_type == 'issue':
                issues.append({
                    'issue_type': item.get('issue_type', ''),
                    'confidence': float(item.get('confidence', 0.0))
                })
            elif item_type == 'user_tag':
                tags.append(item.get('tag_name', ''))
            elif item_type == 'image_label':
                label = item.get('category_name', '')
                if label:
                    labels.append(label)
        
        return issues, tags, labels
    
    extracted = df['metadata_items'].apply(parse_items)
    df['issues'] = extracted.apply(lambda x: x[0])
    df['tags'] = extracted.apply(lambda x: x[1])
    df['labels'] = extracted.apply(lambda x: x[2])
    df['label'] = df['labels'].apply(lambda x: x[0] if x else 'unknown')
    
    # Statistics
    total_issues = sum(len(issues) for issues in df['issues'])
    total_tags = sum(len(tags) for tags in df['tags'])
    
    print(f"  ✓ Parsed metadata")
    print(f"    Total issues detected: {total_issues:,}")
    print(f"    Total user tags: {total_tags:,}")
    print(f"    Unique classes: {df['label'].nunique()}")
    print(f"    Uniqueness range: [{df['uniqueness_score'].min():.3f}, {df['uniqueness_score'].max():.3f}]")
    
    return df


def apply_cleaning_policy(df, policy):
    """Apply cleaning policy to dataset."""
    print_header("Step 3: Apply Cleaning Policy")
    
    drop_reasons = defaultdict(list)
    drop_mask = pd.Series([False] * len(df), index=df.index)
    
    # 1. Uniqueness filter
    if 'uniqueness_threshold' in policy:
        threshold = policy['uniqueness_threshold']
        low_uniq_mask = df['uniqueness_score'] < threshold
        drop_mask |= low_uniq_mask
        
        for idx in df[low_uniq_mask].index:
            fname = df.loc[idx, 'file_name']
            score = df.loc[idx, 'uniqueness_score']
            drop_reasons[fname].append(f"low_uniqueness<{threshold} (score={score:.3f})")
        
        print(f"  🎯 Uniqueness < {threshold}: {low_uniq_mask.sum():,} images")
    
    # 2. Cluster deduplication
    if policy.get('dedupe_by_cluster', False):
        print("  🔄 Deduplicating by cluster...")
        dup_count = 0
        
        for cluster_id, group in df[df['cluster_id'] != -1].groupby('cluster_id'):
            if len(group) <= 1:
                continue
            
            max_idx = group['uniqueness_score'].idxmax()
            dup_indices = group.index[group.index != max_idx]
            
            drop_mask.loc[dup_indices] = True
            for idx in dup_indices:
                fname = df.loc[idx, 'file_name']
                drop_reasons[fname].append(f"duplicate_in_cluster_{cluster_id}")
                dup_count += 1
        
        print(f"  🔄 Cluster duplicates: {dup_count:,} images")
    
    # 3. Issue filters
    for issue_rule in policy.get('drop_issues', []):
        issue_type = issue_rule['issue_type']
        min_conf = issue_rule['min_confidence']
        
        issue_mask = df['issues'].apply(
            lambda issues: any(
                i['issue_type'] == issue_type and i['confidence'] >= min_conf
                for i in issues
            )
        )
        
        drop_mask |= issue_mask
        
        for idx in df[issue_mask].index:
            fname = df.loc[idx, 'file_name']
            drop_reasons[fname].append(f"issue_{issue_type}")
        
        print(f"  🔍 Issue {issue_type} (conf>={min_conf}): {issue_mask.sum():,} images")
    
    # 4. Tag filters
    drop_tags = set(policy.get('drop_tags', []))
    if drop_tags:
        tag_mask = df['tags'].apply(lambda tags: any(t in drop_tags for t in tags))
        drop_mask |= tag_mask
        
        for idx in df[tag_mask].index:
            fname = df.loc[idx, 'file_name']
            matching_tags = [t for t in df.loc[idx, 'tags'] if t in drop_tags]
            drop_reasons[fname].append(f"user_tag: {', '.join(matching_tags)}")
        
        print(f"  🏷️  User tags: {tag_mask.sum():,} images")
    
    # Split into keep/drop
    keep_df = df[~drop_mask].copy()
    drop_df = df[drop_mask].copy()
    
    # Add reasons to drop_df
    drop_df['drop_reasons'] = drop_df['file_name'].map(
        lambda x: ' | '.join(drop_reasons.get(x, []))
    )
    
    # Statistics
    print(f"\n  ✅ Results:")
    print(f"    Kept: {len(keep_df):,} ({100*len(keep_df)/len(df):.1f}%)")
    print(f"    Dropped: {len(drop_df):,} ({100*len(drop_df)/len(df):.1f}%)")
    
    # Multi-reason analysis
    multi_reason = sum(1 for reasons in drop_reasons.values() if len(reasons) > 1)
    if multi_reason > 0:
        print(f"    Multi-reason drops: {multi_reason:,} ({100*multi_reason/len(drop_df):.1f}% of drops)")
    
    return keep_df, drop_df, drop_reasons


def export_clean_dataset(keep_df, extract_dir, export_dir, policy, include_images):
    """Export clean dataset for teammates."""
    print_header("Step 4: Export Clean Dataset")
    
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save metadata
    print(f"  💾 Saving clean metadata ({len(keep_df):,} images)...")
    keep_df.to_json(export_dir / "metadata.json", index=False)
    print(f"    ✓ metadata.json")
    
    # 2. Copy images
    if include_images:
        images_src = extract_dir / "images"
        images_dst = export_dir / "images"
        
        if not images_src.exists():
            print(f"  ⚠️  Warning: Source images not found at {images_src}")
            print(f"    Run with --include-images to download from Visual Layer")
        else:
            images_dst.mkdir(exist_ok=True)
            
            print(f"  📁 Copying {len(keep_df):,} clean images...")
            copied = 0
            missing = 0
            
            for fname in tqdm(keep_df['file_name'], desc="    Copying images"):
                src = images_src / fname
                dst = images_dst / fname
                
                # Create subdirectories if needed
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                if src.exists():
                    shutil.copy2(src, dst)
                    copied += 1
                else:
                    missing += 1
            
            print(f"    ✓ Copied: {copied:,} images")
            if missing > 0:
                print(f"    ⚠️  Missing: {missing:,} images")
    
    # 3. Create README
    readme_content = f"""# Clean ImageNet-1K Dataset

## Overview
Clean subset of ImageNet-1K training set for noise injection experiments.

**⚠️ Original Visual Layer dataset NOT modified - this is a filtered export.**

## Statistics
- **Clean Images**: {len(keep_df):,}
- **Classes**: {keep_df['label'].nunique()}
- **Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Policy Applied
- Uniqueness threshold: {policy.get('uniqueness_threshold', 'N/A')}
- Cluster deduplication: {policy.get('dedupe_by_cluster', False)}
- Issue filters: {len(policy.get('drop_issues', []))}
- Tag filters: {len(policy.get('drop_tags', []))}

## Contents
- `metadata.json` - Clean image metadata with Visual Layer signals
- `images/` - Clean image files {'(included)' if include_images else '(not included)'}
- `cleaning_policy.yaml` - Policy configuration
- `README.md` - This file

## Visual Layer Signals
Each image includes:
- `file_name` - Original filename
- `file_path` - Original path
- `media_id` - Visual Layer internal ID
- `cluster_id` - Visual similarity cluster
- `uniqueness_score` - Visual distinctiveness (0-1)
- `label` - Image class/category
- `issues` - Detected quality issues
- `tags` - User-applied tags

## Usage

### Load Clean Dataset
```python
import pandas as pd

df = pd.read_json('metadata.json')
print(f"Clean dataset: {{len(df):,}} images")
print(f"Classes: {{df['label'].nunique()}}")
```

### For Noise Injection
```python
import numpy as np

# Inject 20% symmetric noise
noise_rate = 0.2
n_corrupt = int(len(df) * noise_rate)
corrupt_idx = np.random.choice(len(df), n_corrupt, replace=False)

df_noisy = df.copy()
all_labels = df['label'].unique()

for idx in corrupt_idx:
    original = df_noisy.loc[idx, 'label']
    new_label = np.random.choice([l for l in all_labels if l != original])
    df_noisy.loc[idx, 'label'] = new_label

df_noisy.to_json('noisy_20pct.json')
```

## Training
```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, metadata_path, images_root, transform=None):
        self.df = pd.read_json(metadata_path)
        self.images_root = Path(images_root)
        self.transform = transform
        
        # Build label mapping
        self.label_to_idx = {{l: i for i, l in enumerate(sorted(self.df['label'].unique()))}}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.images_root / row['file_name']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label_idx = self.label_to_idx[row['label']]
        return img, label_idx

# Usage
dataset = ImageNetDataset('metadata.json', 'images/')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Questions?
Contact your dataset curator for questions about this cleaned version.
"""
    
    with open(export_dir / "README.md", 'w') as f:
        f.write(readme_content)
    print(f"    ✓ README.md")
    
    # 4. Copy policy
    policy_src = Path(args.policy)
    if policy_src.exists():
        shutil.copy2(policy_src, export_dir / "cleaning_policy.yaml")
        print(f"    ✓ cleaning_policy.yaml")
    
    print(f"\n  ✅ Clean dataset exported to: {export_dir}")
    
    return export_dir


def create_zip_archive(export_dir, output_path=None):
    """Create ZIP archive of clean dataset."""
    print_header("Step 5: Create ZIP Archive")
    
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = export_dir.parent / f"clean_imagenet_{timestamp}.zip"
    
    output_path = Path(output_path)
    
    print(f"  📦 Creating ZIP: {output_path.name}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(list(export_dir.rglob('*')), desc="    Compressing"):
            if file_path.is_file():
                arcname = file_path.relative_to(export_dir)
                zipf.write(file_path, arcname)
    
    zip_size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✅ Created: {output_path.name} ({zip_size_gb:.2f} GB)")
    
    return output_path


def save_cleaning_report(keep_df, drop_df, policy, drop_reasons, output_dir):
    """Save cleaning report and statistics."""
    print_header("Step 6: Generate Cleaning Report")
    
    output_dir = Path(output_dir)
    
    # Summary statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(keep_df) + len(drop_df),
        'kept': len(keep_df),
        'dropped': len(drop_df),
        'drop_rate': len(drop_df) / (len(keep_df) + len(drop_df)),
        'policy': policy,
        'drop_by_reason': {},
    }
    
    # Count by reason
    from collections import Counter
    reason_counts = Counter()
    for reasons in drop_reasons.values():
        for reason in reasons:
            base_reason = reason.split('(')[0].strip().split('<')[0]
            reason_counts[base_reason] += 1
    
    summary['drop_by_reason'] = dict(reason_counts)
    
    # Save summary
    summary_path = output_dir / "cleaning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Saved: {summary_path.name}")
    
    # Save manifests
    keep_manifest = output_dir / "keep_filenames.txt"
    drop_manifest = output_dir / "drop_filenames.txt"
    
    keep_df['file_name'].to_csv(keep_manifest, index=False, header=False)
    drop_df['file_name'].to_csv(drop_manifest, index=False, header=False)
    
    print(f"  ✓ Saved: {keep_manifest.name} ({len(keep_df):,} files)")
    print(f"  ✓ Saved: {drop_manifest.name} ({len(drop_df):,} files)")
    
    return summary_path


def main():
    """Main execution function."""
    global args
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    
    print("=" * 70)
    print("ImageNet-1K Dataset Cleaning with Visual Layer")
    print("=" * 70)
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Policy: {args.policy}")
    print(f"Output: {args.output}")
    print(f"Export: {args.export}")
    print(f"Include images: {not args.no_images}")
    print(f"Dry-run auth: {args.dry_run_auth}")
    print("=" * 70)
    
    try:
        # Initialize Visual Layer client
        print("\n🔐 Initializing Visual Layer API client...")
        client = VisualLayerAPIClient()
        
        # Test connection
        print("\n🧪 Testing connection...")
        if not client.test_connection(args.dataset_id):
            print("\n❌ Connection test failed. Please check:")
            print("  1. Dataset ID is correct")
            print("  2. VL_API_KEY and VL_API_SECRET are set")
            print("  3. You have access to this dataset")
            return 1

        if args.dry_run_auth:
            print("\n✅ Dry-run auth passed")
            print("  JWT generation and dataset connectivity are valid.")
            print("  Skipping policy load, export, parsing, and cleaning.")
            return 0
        
        # Load policy
        policy = load_policy(args.policy)
        
        # Export from Visual Layer
        extract_dir = export_from_visual_layer(
            client, args.dataset_id, args.output, not args.no_images
        )
        
        # Parse metadata
        df = parse_metadata(extract_dir)
        
        # Apply cleaning policy
        keep_df, drop_df, drop_reasons = apply_cleaning_policy(df, policy)
        
        # Export clean dataset
        export_dir = export_clean_dataset(
            keep_df, extract_dir, args.export, policy, not args.no_images
        )
        
        # Create ZIP
        zip_path = create_zip_archive(export_dir)
        
        # Save report
        save_cleaning_report(keep_df, drop_df, policy, drop_reasons, Path(args.output))
        
        # Copy to VM if requested
        if args.vm_path:
            print_header("Step 7: Copy to VM")
            vm_dest = Path(args.vm_path) / zip_path.name
            print(f"  📦 Copying to: {vm_dest}")
            
            vm_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, vm_dest)
            
            print(f"  ✅ Copied to VM: {vm_dest}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("✅ Cleaning Complete!")
        print("=" * 70)
        print(f"📊 Results:")
        print(f"  • Original: {len(df):,} images")
        print(f"  • Clean: {len(keep_df):,} images ({100*len(keep_df)/len(df):.1f}%)")
        print(f"  • Removed: {len(drop_df):,} images ({100*len(drop_df)/len(df):.1f}%)")
        print(f"\n📦 Outputs:")
        print(f"  • Clean dataset: {export_dir}")
        print(f"  • ZIP file: {zip_path}")
        if args.vm_path:
            print(f"  • VM copy: {Path(args.vm_path) / zip_path.name}")
        print("\n📤 Share with teammates:")
        if args.vm_path:
            print(f"  {Path(args.vm_path) / zip_path.name}")
        else:
            print(f"  {zip_path}")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_python_cache(script_dir)


if __name__ == '__main__':
    sys.exit(main())
