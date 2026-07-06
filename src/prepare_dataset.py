import os
import sys
import shutil
import random
import yaml
import numpy as np
import cv2
from tqdm import tqdm

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def clean_directories(base_dir):
    """Clears existing dataset directories to ensure a clean generation."""
    splits = ['train', 'val', 'test']
    for split in splits:
        img_path = os.path.join(base_dir, "images", split)
        lbl_path = os.path.join(base_dir, "labels", split)
        
        # Clean images
        if os.path.exists(img_path):
            for file in os.listdir(img_path):
                file_full = os.path.join(img_path, file)
                if os.path.isfile(file_full):
                    os.remove(file_full)
                    
        # Clean labels
        if os.path.exists(lbl_path):
            for file in os.listdir(lbl_path):
                file_full = os.path.join(lbl_path, file)
                if os.path.isfile(file_full):
                    os.remove(file_full)
                    
    print("🧹 Cleared old dataset files from images and labels directories.")

def generate_rough_ellipse_points(cx, cy, rx, ry, num_points=30, noise_level=0.15):
    """
    Generates points for an ellipse with added noise to make the boundary look rough and jagged.
    
    Args:
        cx, cy: Center coordinates of the ellipse.
        rx, ry: Radii along the x and y axes.
        num_points: Number of vertices in the polygon.
        noise_level: Scale of the random perturbation applied to the radius.
        
    Returns:
        numpy.ndarray: Vertex points of the rough shape.
    """
    points = []
    for i in range(num_points):
        # Angle from 0 to 2*pi
        theta = (i / num_points) * 2 * np.pi
        # Standard ellipse radius formula (parametric)
        x_base = rx * np.cos(theta)
        y_base = ry * np.sin(theta)
        
        # Calculate current radius from center
        r = np.sqrt(x_base**2 + y_base**2)
        # Add random noise to the radius
        noise = 1 + np.random.uniform(-noise_level, noise_level)
        r_noisy = r * noise
        
        # Compute final coordinate relative to center
        # We preserve the direction (angle theta)
        angle = np.arctan2(y_base, x_base)
        px = cx + r_noisy * np.cos(angle)
        py = cy + r_noisy * np.sin(angle)
        
        points.append([int(px), int(py)])
        
    return np.array(points, dtype=np.int32)

def create_synthetic_road_image():
    """
    Generates a synthetic road image with an asphalt texture, lane lines, 
    and random rough-edged potholes.
    
    Returns:
        img: The generated OpenCV image (640x640x3).
        labels: List of bounding boxes in YOLO format [(class_id, x_c, y_c, w, h)].
    """
    # 1. Create a dark gray asphalt background (640x640)
    base_color = 80
    img = np.full((640, 640, 3), base_color, dtype=np.uint8)
    
    # Add random noise/grain to simulate rough asphalt texture
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. Draw white/yellow dashed lane markings
    lane_color = (220, 220, 220)  # BGR white
    lane_width = 8
    lane_length = 40
    gap_length = 30
    
    for y in range(0, 640, lane_length + gap_length):
        cv2.rectangle(img, (320 - lane_width // 2, y), (320 + lane_width // 2, y + lane_length), lane_color, -1)
        
    # Also draw side solid lines to make it look like a highway lane
    cv2.line(img, (60, 0), (60, 640), (200, 200, 200), 4)  # Left solid line
    cv2.line(img, (580, 0), (580, 640), (200, 200, 200), 4) # Right solid line
    
    # 3. Add 2-5 potholes per image
    num_potholes = random.randint(2, 5)
    labels = []
    
    for _ in range(num_potholes):
        # Keep potholes within the road markings (avoid edges too close to the boundary)
        cx = random.randint(100, 540)
        cy = random.randint(50, 590)
        
        # Dimensions of the pothole
        rx = random.randint(25, 75)
        ry = random.randint(15, 45)
        
        # Generate rough edge vertices
        pts = generate_rough_ellipse_points(cx, cy, rx, ry, num_points=36, noise_level=0.18)
        
        # Draw a slightly larger, lighter rim for depth (shadow/cracked edge effect)
        rim_color = (np.random.randint(45, 55), np.random.randint(45, 55), np.random.randint(45, 55))
        cv2.drawContours(img, [pts], -1, rim_color, thickness=3)
        
        # Draw the main dark cavity of the pothole
        cavity_color = (np.random.randint(20, 35), np.random.randint(20, 35), np.random.randint(20, 35))
        cv2.fillPoly(img, [pts], cavity_color)
        
        # Optional: Add small crack lines extending from the pothole
        for _ in range(random.randint(1, 3)):
            # Pick a random point on the boundary
            start_pt = pts[random.randint(0, len(pts)-1)]
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(10, 30)
            end_pt = (int(start_pt[0] + length * np.cos(angle)), int(start_pt[1] + length * np.sin(angle)))
            cv2.line(img, tuple(start_pt), end_pt, (40, 40, 40), 1)
            
        # 4. Calculate bounding box of the actual drawn points to ensure exact labeling
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        
        # Clamp bounding box coordinates to image dimensions
        min_x = max(0, min_x)
        max_x = min(640, max_x)
        min_y = max(0, min_y)
        max_y = min(640, max_y)
        
        # Calculate YOLO box coordinates (normalized)
        width_px = max_x - min_x
        height_px = max_y - min_y
        
        x_center = (min_x + width_px / 2.0) / 640.0
        y_center = (min_y + height_px / 2.0) / 640.0
        w_norm = width_px / 640.0
        h_norm = height_px / 640.0
        
        # Class ID is 0 for potholes
        labels.append((0, x_center, y_center, w_norm, h_norm))
        
    return img, labels

def save_data(img, labels, base_dir, split_name, img_id):
    """Saves the image and its corresponding label text file in the folder structure."""
    img_filename = f"road_{img_id:04d}.jpg"
    lbl_filename = f"road_{img_id:04d}.txt"
    
    img_path = os.path.join(base_dir, "images", split_name, img_filename)
    lbl_path = os.path.join(base_dir, "labels", split_name, lbl_filename)
    
    # Save Image
    cv2.imwrite(img_path, img)
    
    # Save YOLO Label File
    with open(lbl_path, "w") as f:
        for lbl in labels:
            class_id, x, y, w, h = lbl
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def validate_dataset(base_dir, splits):
    """
    Validates that:
    1. Bounding boxes are normalized between 0 and 1.
    2. Class IDs are only 0.
    3. Every image file has a matching label file.
    
    Returns:
        dict: Summary statistics and count of issues.
    """
    issues_found = 0
    total_potholes = 0
    total_images = 0
    
    for split in splits:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        
        img_files = set([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')])
        lbl_files = set([os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')])
        
        # Check matching files
        mismatched_imgs = img_files - lbl_files
        mismatched_lbls = lbl_files - img_files
        
        if mismatched_imgs:
            print(f"❌ Mismatch: Images without label in '{split}': {mismatched_imgs}")
            issues_found += len(mismatched_imgs)
        if mismatched_lbls:
            print(f"❌ Mismatch: Labels without image in '{split}': {mismatched_lbls}")
            issues_found += len(mismatched_lbls)
            
        # Check label contents
        for file_name in lbl_files:
            total_images += 1
            lbl_path = os.path.join(lbl_dir, f"{file_name}.txt")
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    total_potholes += 1
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"❌ Malformed annotation in {lbl_path}: '{line.strip()}'")
                        issues_found += 1
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                    except ValueError:
                        print(f"❌ Non-numeric values in {lbl_path}: '{line.strip()}'")
                        issues_found += 1
                        continue
                        
                    # Check class ID
                    if class_id != 0:
                        print(f"❌ Invalid Class ID {class_id} in {lbl_path}. Must be 0.")
                        issues_found += 1
                        
                    # Check bounding box bounds
                    for val_name, val in [('x', x), ('y', y), ('w', w), ('h', h)]:
                        if not (0.0 <= val <= 1.0):
                            print(f"❌ Out-of-bounds coordinate {val_name}={val} in {lbl_path}")
                            issues_found += 1
                            
    return {
        'issues': issues_found,
        'total_images': total_images,
        'total_potholes': total_potholes
    }

def main():
    print("🛣️  Generating Synthetic Pothole Dataset...")
    set_seed(42)
    
    base_dir = "dataset"
    
    # Ensure setup.py ran and folder structure exists, otherwise create it
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)
        
    # Clean directories first to ensure no old files pollute statistics
    clean_directories(base_dir)
    
    total_images = 300
    
    # 70% Train, 20% Val, 10% Test
    train_count = 210
    val_count = 60
    test_count = 30
    
    # Generate splits mapping
    splits = []
    for i in range(1, train_count + 1):
        splits.append(('train', i))
    for i in range(train_count + 1, train_count + val_count + 1):
        splits.append(('val', i))
    for i in range(train_count + val_count + 1, total_images + 1):
        splits.append(('test', i))
        
    # Generate and save images and labels
    for split_name, img_id in tqdm(splits, desc="Generating Images"):
        img, labels = create_synthetic_road_image()
        save_data(img, labels, base_dir, split_name, img_id)
        
    print("✅ Synthetic image generation complete.")
    
    # Create dataset/data.yaml
    abs_dataset_path = os.path.abspath(base_dir).replace("\\", "/") # Normalize backslashes for PyYAML/YOLO
    
    yaml_data = {
        'path': abs_dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['pothole']
    }
    
    yaml_path = os.path.join(base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False)
    print(f"✅ Created {yaml_path} with absolute path: {abs_dataset_path}")
    
    # Validate Annotations
    print("\n🔍 Validating Dataset Annotations...")
    validation_results = validate_dataset(base_dir, ['train', 'val', 'test'])
    
    if validation_results['issues'] == 0:
        print("✅ Annotation validation passed! All coordinates are between 0 and 1, class IDs are 0, and all image-label pairs match.")
    else:
        print(f"❌ Annotation validation failed with {validation_results['issues']} issues. Check stdout above.")
        
    # Print Statistics
    total_potholes = validation_results['total_potholes']
    avg_potholes = total_potholes / total_images if total_images > 0 else 0
    
    print("\n📊 Dataset Statistics:")
    print(f"   Training images:   {train_count}")
    print(f"   Validation images: {val_count}")
    print(f"   Test images:       {test_count}")
    print(f"   Total potholes:    {total_potholes}")
    print(f"   Avg potholes/img:  {avg_potholes:.2f}")

if __name__ == "__main__":
    main()
