import os

def sanitize_filenames(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            # only process if there are spaces or commas
            if " " in name or "," in name:
                old_path = os.path.join(dirpath, name)
                # replace spaces with underscores, remove commas
                new_name = name.replace(" ", "_").replace(",", "")
                new_path = os.path.join(dirpath, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} → {new_path}")
                except Exception as e:
                    print(f"Failed to rename {old_path}: {e}")

if __name__ == "__main__":
    folder = r"C:\Users\Formu\Downloads\OneDrive_2025-07-30\SustainBuddy Knowledge"
    if os.path.isdir(folder):
        sanitize_filenames(folder)
    else:
        print(f"Error: folder not found: {folder}")
