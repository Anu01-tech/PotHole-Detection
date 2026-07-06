import os
import sys
import shutil

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main():
    train_img_dir = "dataset/images/train"
    train_lbl_dir = "dataset/labels/train"
    backup_img_dir = "dataset/images/train_backup"
    backup_lbl_dir = "dataset/labels/train_backup"
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Training directory '{train_img_dir}' not found. Run setup_real_dataset.py first.")
        sys.exit(1)
        
    # Get all training images
    img_files = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith('.jpg')])
    total_images = len(img_files)
    
    print(f"📊 Current training dataset size: {total_images} images.")
    
    target_count = 600
    if total_images <= target_count:
        print("ℹ️  Dataset is already optimized (less than or equal to 600 images). Skipping subset creation.")
        return
        
    print(f"⚡ Optimizing dataset for CPU training: Reducing from {total_images} to {target_count} images...")
    
    # Create backup directories
    os.makedirs(backup_img_dir, exist_ok=True)
    os.makedirs(backup_lbl_dir, exist_ok=True)
    
    # Move the extra images and labels to the backup folder
    files_to_backup = img_files[target_count:]
    
    moved_count = 0
    for file_name in files_to_backup:
        base_name = os.path.splitext(file_name)[0]
        
        # Paths
        src_img = os.path.join(train_img_dir, file_name)
        dst_img = os.path.join(backup_img_dir, file_name)
        
        src_lbl = os.path.join(train_lbl_dir, f"{base_name}.txt")
        dst_lbl = os.path.join(backup_lbl_dir, f"{base_name}.txt")
        
        # Move image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
            moved_count += 1
            
        # Move label
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
            
    print(f"✅ Moved {moved_count} images & matching labels to backup folder to speed up CPU training.")
    print(f"📊 New training dataset size: {len(os.listdir(train_img_dir))} images.")
    print("🚀 Ready for optimized 10-epoch training.")

if __name__ == "__main__":
    main()
