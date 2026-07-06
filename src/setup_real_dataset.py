import os
import sys
import shutil
import urllib.request
import zipfile

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def clean_old_dataset(base_dir):
    """Cleans train, val, and test directories in dataset."""
    splits = ['train', 'val', 'test']
    for split in splits:
        for folder in ['images', 'labels']:
            path = os.path.join(base_dir, folder, split)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    print("🧹 Cleared existing dataset files.")

def main():
    url = "https://learnopencv.s3.us-west-2.amazonaws.com/pothole_dataset.zip"
    zip_path = "pothole_dataset.zip"
    extract_temp_dir = "real_dataset_temp"
    dataset_dir = "dataset"
    
    # 1. Download if not exists
    if not os.path.exists(zip_path):
        print("📥 Downloading real pothole dataset from LearnOpenCV (S3)...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            sys.exit(1)
            
    # 2. Extract
    print("📦 Extracting dataset (this may take a few seconds)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_temp_dir)
    print("✅ Extraction complete.")
    
    # Identify the extracted folder name
    src_pothole_dir = os.path.join(extract_temp_dir, "pothole_dataset")
    if not os.path.exists(src_pothole_dir):
        src_pothole_dir = extract_temp_dir

    # Clean the current dataset directory first
    clean_old_dataset(dataset_dir)
    
    # Map splits: Roboflow uses 'valid' instead of 'val'
    splits = ['train', 'val', 'test']
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    print("🚚 Rearranging dataset splits...")
    for split in splits:
        # Check for 'valid' folder name if split is 'val'
        src_folder_name = "valid" if split == "val" else split
        
        src_img_dir = os.path.join(src_pothole_dir, "images", src_folder_name)
        src_lbl_dir = os.path.join(src_pothole_dir, "labels", src_folder_name)
        
        dst_img_dir = os.path.join(dataset_dir, "images", split)
        dst_lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        # Check if source directory exists
        if os.path.exists(src_img_dir):
            for file in os.listdir(src_img_dir):
                shutil.copy2(os.path.join(src_img_dir, file), os.path.join(dst_img_dir, file))
                counts[split] += 1
                
        if os.path.exists(src_lbl_dir):
            for file in os.listdir(src_lbl_dir):
                shutil.copy2(os.path.join(src_lbl_dir, file), os.path.join(dst_lbl_dir, file))
                
    # If test split is missing or empty, create one by splitting a portion of val/train
    if counts['test'] == 0:
        print("ℹ️  No test split found in source. Creating test split from validation data...")
        # Move a portion (e.g., 20%) of validation data to test
        val_img_dir = os.path.join(dataset_dir, "images", "val")
        val_lbl_dir = os.path.join(dataset_dir, "labels", "val")
        
        test_img_dir = os.path.join(dataset_dir, "images", "test")
        test_lbl_dir = os.path.join(dataset_dir, "labels", "test")
        
        val_files = [os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
        # Move 1 in 5 files
        test_files = val_files[::5]
        
        for file_base in test_files:
            shutil.move(os.path.join(val_img_dir, f"{file_base}.jpg"), os.path.join(test_img_dir, f"{file_base}.jpg"))
            counts['val'] -= 1
            counts['test'] += 1
            
            val_lbl_path = os.path.join(val_lbl_dir, f"{file_base}.txt")
            if os.path.exists(val_lbl_path):
                shutil.move(val_lbl_path, os.path.join(test_lbl_dir, f"{file_base}.txt"))
                
    # Create dataset/data.yaml with absolute paths
    abs_dataset_path = os.path.abspath(dataset_dir).replace("\\", "/")
    yaml_data = {
        'path': abs_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['pothole']
    }
    
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        import yaml
        yaml.safe_dump(yaml_data, f, default_flow_style=False)
        
    print(f"✅ Created {yaml_path} with absolute path.")
    
    # Cleanup temporary directories and download zip
    print("🧹 Cleaning up temporary extraction files...")
    shutil.rmtree(extract_temp_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("✅ Cleanup complete.")
    
    # Print statistics
    print("\n📊 Real Dataset Statistics:")
    print(f"   Training images:   {counts['train']}")
    print(f"   Validation images: {counts['val']}")
    print(f"   Test images:       {counts['test']}")
    print(f"   Total Real Images: {counts['train'] + counts['val'] + counts['test']}")
    print("🎉 Real-world dataset setup complete! Ready to start training.")

if __name__ == "__main__":
    main()
