import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def count_files(directory, extensions):
    """Count files with specific extensions in a directory."""
    if not os.path.exists(directory):
        return 0, []
    
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*.{ext}"))
    
    return len(files), files

def validate_data():
    """Validate all data directories."""
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60 + "\n")

    text_dir = os.getenv("TEXT_DATA_DIR", "./data/sample_document")
    image_dir = os.getenv("IMAGE_DATA_DIR", "./data/sample_image")
    csv_dir = os.getenv("CSV_DATA_DIR", "./data/csv")
    
    all_valid = True
    
    print("TEXT DOCUMENTS:")
    text_count, text_files = count_files(text_dir, ["pdf", "txt", "docx"])
    print(f"   Directory: {text_dir}")
    print(f"   Status: {' READY' if text_count > 0 else ' EMPTY'}")
    print(f"   Files found: {text_count}")
    if text_count > 0:
        print(f"   Sample files: {', '.join([f.name for f in list(text_files)[:3]])}")
    all_valid = all_valid and (text_count > 0)
    print()
    
    # Check images
    print("IMAGE FILES:")
    image_count, image_files = count_files(image_dir, ["pdf", "jpg", "jpeg", "png", "tiff", "bmp"])
    print(f"   Directory: {image_dir}")
    print(f"   Status: {' READY' if image_count > 0 else ' EMPTY'}")
    print(f"   Files found: {image_count}")
    if image_count > 0:
        print(f"   Sample files: {', '.join([f.name for f in list(image_files)[:3]])}")
    all_valid = all_valid and (image_count > 0)
    print()
    
    # Check CSVs
    print(" CSV FILES:")
    csv_count, csv_files = count_files(csv_dir, ["csv"])
    print(f"   Directory: {csv_dir}")
    print(f"   Status: {' READY' if csv_count > 0 else ' EMPTY'}")
    print(f"   Files found: {csv_count}")
    if csv_count > 0:
        print(f"   Sample files: {', '.join([f.name for f in list(csv_files)[:3]])}")
    all_valid = all_valid and (csv_count > 0)
    print()
    
    # Summary
    print("="*60)
    total = text_count + image_count + csv_count
    print(f"TOTAL FILES: {total}")
    print("="*60)
    
    if all_valid:
        print("\n ALL DATA VALIDATED - Ready to process!")
        print("\nNext step:")
        print("   python orchestrate.py --all")
        return 0
    else:
        print("\n VALIDATION FAILED - Some directories are empty")
        print("\nPlease ensure sample data is in:")
        print(f"   - {text_dir}")
        print(f"   - {image_dir}")
        print(f"   - {csv_dir}")
        return 1

if __name__ == "__main__":
    exit(validate_data())
