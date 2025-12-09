import os
import gdown
import zipfile
import shutil
import time

# ----------------------------
# DeepFashion FILE IDs
# ----------------------------
FILES = {
    "deepfashion_category.zip": "1f36GdNyBom4GXWvfbhNHxP5TQ_G2SB-r",
    "deepfashion_retrieval.zip": "1KzQ-tO5PrUgrjyjrsny4Arsl4CLN2z6G"
}

RAW_DIR = "data/raw"


# ----------------------------
# 1) DOWNLOAD ZIP FILE
# ----------------------------
def download_file(file_id, output_path):
    print(f"⬇️  Downloading {output_path} ...")

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False, fuzzy=True)

    print(f"✅ Downloaded: {output_path}\n")


# ----------------------------
# 2) CHECK ZIP VALIDITY
# ----------------------------
def is_zip_valid(zip_path):
    """Return True if zip is valid, False if corrupted."""
    if not os.path.exists(zip_path):
        return False

    # Suspicious small file (< 10MB → likely corrupted)
    if os.path.getsize(zip_path) < 10_000_000:
        print(f"⚠️  ZIP suspiciously small: {zip_path}")
        return False

    # Check basic zip file structure
    if not zipfile.is_zipfile(zip_path):
        print(f"❌ Not a valid ZIP file structure: {zip_path}")
        return False

    # Try opening zip contents
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            test = z.testzip()
            if test is not None:
                print(f"❌ ZIP contains a bad file: {test}")
                return False
    except:
        print(f"❌ ZIP cannot be read: {zip_path}")
        return False

    return True


# ----------------------------
# 3) SMART DOWNLOAD (AUTO FIX)
# ----------------------------
def smart_download(filename, file_id):
    zip_path = os.path.join(RAW_DIR, filename)

    # If exists → check if valid
    if os.path.exists(zip_path):
        print(f"📦 Checking existing ZIP: {zip_path}")

        if is_zip_valid(zip_path):
            print(f"✔ Valid ZIP found — skipping download.\n")
            return zip_path
        else:
            print(f"❌ ZIP is corrupted — removing and redownloading.\n")
            os.remove(zip_path)

    # Attempt up to 3 times
    for attempt in range(2):
        print(f"🔄 Download Attempt {attempt + 1}/2 for {filename}")
        download_file(file_id, zip_path)

        if is_zip_valid(zip_path):
            print(f"🎉 ZIP downloaded & verified: {filename}\n")
            return zip_path

        print("⚠️ Downloaded ZIP still corrupted, retrying...")
        os.remove(zip_path)
        time.sleep(1)

    print(f"❌ FAILED: Could not download a valid ZIP after 3 attempts: {filename}")
    exit(1)


# ----------------------------
# UNZIP
# ----------------------------
def unzip_file(zip_path, target_dir):
    print(f"📦 Unzipping {zip_path} → {target_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(target_dir)
        print(f"✅ Extracted to {target_dir}\n")
    except Exception as e:
        print(f"❌ Extraction error for {zip_path}: {e}")
        exit(1)


# ----------------------------
# FIX NESTED FOLDER
# ----------------------------
def fix_nested_folder(root_folder):
    nested = os.path.join(root_folder, os.path.basename(root_folder))

    if os.path.isdir(nested):
        print(f"🔧 Fixing nested folder at: {nested}")
        for item in os.listdir(nested):
            shutil.move(os.path.join(nested, item), os.path.join(root_folder, item))
        shutil.rmtree(nested)
        print(f"✅ Nested folder fixed → {root_folder}\n")
    else:
        print(f"✔ No nested folder in {root_folder}\n")


# ----------------------------
# EXTRACT inner Img/img.zip
# ----------------------------
def unzip_internal_img_zip(root_folder):
    img_zip = os.path.join(root_folder, "Img", "img.zip")
    if os.path.exists(img_zip):
        print(f"📷 Extracting inner image archive: {img_zip}")
        unzip_file(img_zip, os.path.join(root_folder, "Img"))
        os.remove(img_zip)
        print(f"🧹 Removed img.zip\n")
    else:
        print(f"✔ No internal img.zip in {root_folder}\n")


# ----------------------------
# MAIN SETUP
# ----------------------------
def main():
    print("\n=== SnapStyle DeepFashion Dataset Setup (Smart Mode + Auto ZIP Check) ===\n")
    os.makedirs(RAW_DIR, exist_ok=True)

    # ---- Step 1: Smart Download with auto-recovery ----
    zip_paths = {}
    for filename, file_id in FILES.items():
        zip_paths[filename] = smart_download(filename, file_id)

    # ---- Step 2: Unzip each dataset ----
    for filename in FILES.keys():
        zip_path = zip_paths[filename]
        extract_dir = os.path.join(RAW_DIR, filename.replace(".zip", ""))

        os.makedirs(extract_dir, exist_ok=True)
        unzip_file(zip_path, extract_dir)
        fix_nested_folder(extract_dir)
        unzip_internal_img_zip(extract_dir)

    print("🎉 All DeepFashion datasets ready in data/raw/\n")


if __name__ == "__main__":
    main()
