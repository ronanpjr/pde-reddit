import os
import shutil
import glob

# Paths
IMAGES_DIR = "/home/ronan/pde-reddit/data/images"

def normalize_name(s):
    # Remove whitespace, parentheses, digits
    s = s.replace(" (1)", "").replace(" (2)", "").replace(" (3)", "")
    s = s.strip()
    if s.endswith("_fil"):
        s = s[:-4]
    if s == "blackpeopletwittere":
        s = "blackpeopletwitter"
    return s

def main():
    print("Scanning images in:", IMAGES_DIR)
    
    # 1. Gather all image files
    all_files = glob.glob(os.path.join(IMAGES_DIR, "**/*"), recursive=True)
    image_files = []
    for f in all_files:
        if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(f)
            
    print(f"Found {len(image_files)} image files.")
    
    # 2. Process and move each image to a staging directory first
    # This avoids moving a file into a directory we are currently walking or deleting.
    staging_dir = "/home/ronan/pde-reddit/data/_staging_images"
    os.makedirs(staging_dir, exist_ok=True)
    
    moved_count = 0
    for f in image_files:
        # Determine the subreddit from the path components relative to IMAGES_DIR
        rel_path = os.path.relpath(f, IMAGES_DIR)
        parts = rel_path.split(os.sep)
        
        # We find the first part that contains our target name
        # E.g. '2meirl4meirl_fil (1)' or '2meirl4meirl_fil'
        sub_name = None
        for part in parts[:-1]: # exclude the filename itself
            cleaned = normalize_name(part)
            if cleaned:
                sub_name = cleaned
                break
                
        if not sub_name:
            print(f"Could not determine subreddit for: {f}")
            continue
            
        # Target path in staging directory
        dest_subdir = os.path.join(staging_dir, sub_name)
        os.makedirs(dest_subdir, exist_ok=True)
        
        filename = os.path.basename(f)
        dest_file = os.path.join(dest_subdir, filename)
        
        # Move the file
        try:
            shutil.move(f, dest_file)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {f} -> {dest_file}: {e}")
            
    print(f"Successfully moved {moved_count} files to staging.")
    
    # 3. Clean up the old images directory entirely
    print("Cleaning up old directories...")
    try:
        shutil.rmtree(IMAGES_DIR)
    except Exception as e:
        print(f"Error deleting old images directory: {e}")
        
    # 4. Rename staging directory to images
    print("Re-establishing clean images directory...")
    try:
        shutil.move(staging_dir, IMAGES_DIR)
        print("Success! Reorganized images successfully.")
    except Exception as e:
        print(f"Error renaming staging directory back to images: {e}")
        
if __name__ == "__main__":
    main()
