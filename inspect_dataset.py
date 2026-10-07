import os
import io
import zipfile
from collections import Counter, defaultdict
from PIL import Image

ZIP_PATH = os.path.expanduser(r"~\Downloads\FMD_DATASET.zip")

def inspect():
    print(f"Opening zip archive: {ZIP_PATH}")
    if not os.path.exists(ZIP_PATH):
        print(f"Error: {ZIP_PATH} not found.")
        return

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        infolist = z.infolist()
        total_entries = len(infolist)
        print(f"Total entries in archive: {total_entries}")

        classes = Counter()
        subdirs = defaultdict(Counter)
        ext_counter = Counter()
        non_image_files = []
        corrupted_files = []
        resolutions = []
        image_formats = Counter()
        
        # Sample or check all images
        for idx, info in enumerate(infolist):
            if info.is_dir():
                continue
            
            parts = info.filename.split('/')
            top_class = parts[0]
            classes[top_class] += 1
            if len(parts) > 2:
                subdirs[top_class][parts[1]] += 1
            
            _, ext = os.path.splitext(info.filename)
            ext_clean = ext.lower()
            ext_counter[ext_clean] += 1
            
            if ext_clean not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                non_image_files.append(info.filename)
                continue
            
            # Read and verify image
            try:
                data = z.read(info)
                # Verify image can be opened and decoded
                with Image.open(io.BytesIO(data)) as img:
                    img_format = img.format
                    image_formats[img_format] += 1
                    w, h = img.size
                    resolutions.append((w, h))
                    img.verify()
                
                # Further verify decoding by re-opening and loading actual pixels for a subset or all
                # Note: verify() only checks headers. To check if bitstream is corrupt:
                with Image.open(io.BytesIO(data)) as img:
                    img.load()
            except Exception as e:
                corrupted_files.append((info.filename, str(e)))

            if (idx + 1) % 2500 == 0:
                print(f"Processed {idx + 1}/{total_entries} items...")

    print("\n" + "="*50)
    print("DATASET INSPECTION RESULTS")
    print("="*50)
    print("\n1. Folder Structure & Classes:")
    for cls, count in classes.items():
        print(f"  - Class '{cls}': {count} files")
        if cls in subdirs:
            for sub, sub_count in subdirs[cls].items():
                print(f"      Subfolder '{sub}': {sub_count} files")

    print("\n2. File Extensions:")
    for ext, count in ext_counter.items():
        print(f"  - {ext or '[no extension]'}: {count}")

    print("\n3. Detected Image Formats (PIL):")
    for fmt, count in image_formats.items():
        print(f"  - {fmt}: {count}")

    print(f"\n4. Non-Image Files: {len(non_image_files)}")
    for f in non_image_files[:10]:
        print(f"  - {f}")
    if len(non_image_files) > 10:
        print(f"  ... and {len(non_image_files) - 10} more.")

    print(f"\n5. Corrupted / Unreadable Files: {len(corrupted_files)}")
    for f, err in corrupted_files[:10]:
        print(f"  - {f}: {err}")
    if len(corrupted_files) > 10:
        print(f"  ... and {len(corrupted_files) - 10} more.")

    if resolutions:
        widths = [w for w, h in resolutions]
        heights = [h for w, h in resolutions]
        common_res = Counter(resolutions).most_common(5)
        print("\n6. Image Resolutions:")
        print(f"  - Total valid images: {len(resolutions)}")
        print(f"  - Min Resolution: {min(widths)}x{min(heights)}")
        print(f"  - Max Resolution: {max(widths)}x{max(heights)}")
        print(f"  - Avg Resolution: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
        print("  - Most Common Resolutions:")
        for res, count in common_res:
            print(f"      {res[0]}x{res[1]}: {count} images")

if __name__ == '__main__':
    inspect()
