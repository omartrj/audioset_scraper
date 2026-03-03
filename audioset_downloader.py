import json
import csv
import time
import random
import subprocess
import logging
import argparse
import os
from pathlib import Path

# ==============================================================================
# INITIAL SETUP
# ==============================================================================

# Target categories: an array of strings (names or IDs)
TARGET_CATEGORIES = ["Human voice", "Human group actions", "Traffic noise, roadway noise"]

# Categories to avoid: drop anything containing these sounds (or their children)
AVOID_CATEGORIES = ["Emergency vehicle", "Music", "Speech synthesizer", "Babbling", "Narrative, monologue"]

# Host operating system: "windows" or "linux"
HOST_OS = "linux"

# Hard cap per root category to ensure the dataset stays balanced
MAX_SAMPLES_PER_CATEGORY = 1000

# Path where your audio will be saved
OUTPUT_DIR = "downloaded_audio"

# Audio Settings
SAMPLE_RATE = 48000
CHANNELS = 1              # 1 = Mono
MIN_DURATION = 8          # Drop segments shorter than this
MAX_DURATION = 10         # Cap maximum length in seconds

# Browser & Download Settings
USE_COOKIES = False       # Enable this if you hit bot limits or age restrictions (requires a browser)
BROWSER = "firefox"       # Options: chrome, firefox, edge, safari, opera, brave
SLEEP_INTERVAL = [1, 3]   # Random pause between downloads (helps fly under the radar)
BATCH_SIZE = 500          # Max downloads per run (useful for chunking)

# Paths to AudioSet metadata
ONTOLOGY_PATH = "ontology.json"
CSV_PATH = "unbalanced_train_segments.csv"

# ==============================================================================
# LOGGING CONFIG
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def get_target_mappings(ontology_path, target_categories):
    """
    Recursively fetch child IDs for the specified target categories.
    Returns a dict {root_id: set_of_all_sub_ids} used to balance the dataset.
    """
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
    except FileNotFoundError:
        logging.error(f"File {ontology_path} not found.")
        return {}, {}

    name_to_id = {item['name']: item['id'] for item in ontology}
    id_to_item = {item['id']: item for item in ontology}

    def get_all_children(node_id):
        children = set([node_id])
        if node_id in id_to_item:
            for c in id_to_item[node_id].get('child_ids', []):
                children.update(get_all_children(c))
        return children

    target_map = {}
    for cat in target_categories:
        cat_id = name_to_id.get(cat, cat)
        if cat_id in id_to_item:
            target_map[cat_id] = get_all_children(cat_id)
        else:
            logging.warning(f"Skipping category (not found in ontology): {cat}")

    return target_map, id_to_item

def get_existing_counts(metadata_path, target_map):
    """
    Reads metadata.csv (if it exists) to count how many downloads we've already
    secured per category. This lets us easily resume interrupted runs.
    """
    root_counts = {r: 0 for r in target_map.keys()}
    if not metadata_path.exists():
        return root_counts
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels = set(row.get('labels_ids', '').split(';'))
                for root, children in target_map.items():
                    if children.intersection(labels):
                        root_counts[root] += 1
    except Exception as e:
        logging.error(f"Failed reading metadata.csv for counts: {e}")
        
    return root_counts

def parse_csv_for_targets(csv_path, target_map, avoid_map, max_samples, min_duration, metadata_path):
    """
    Scans the CSV, drops short clips, drops clips with unwanted labels, 
    and returns candidates respecting the max_samples limit.
    """
    if not Path(csv_path).exists():
        logging.error(f"CSV file not found: {csv_path}")
        return []

    # Flatten valid IDs for fast lookup
    all_valid_ids = set()
    for children in target_map.values():
        all_valid_ids.update(children)
        
    # Flatten avoid IDs
    all_avoid_ids = set()
    if avoid_map:
        for children in avoid_map.values():
            all_avoid_ids.update(children)

    all_candidates = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                # Skip comments and empty lines
                if not row or row[0].startswith('#'):
                    continue
                
                ytid = row[0]
                start_sec = float(row[1])
                end_sec = float(row[2])
                
                # Drop segments that are too short
                if (end_sec - start_sec) < min_duration:
                    continue
                
                labels_str = ",".join(row[3:]).replace('"', '').strip()
                labels_list = [l.strip() for l in labels_str.split(',')]
                
                # Fast set logic: must have a target, but must NOT have an avoided label
                if all_valid_ids.intersection(labels_list):
                    if not all_avoid_ids.intersection(labels_list):
                        all_candidates.append({
                            'ytid': ytid,
                            'start_seconds': start_sec,
                            'end_seconds': end_sec,
                            'labels_ids': labels_list
                        })
    except Exception as e:
        logging.error(f"Error parsing the CSV: {e}")
        return []

    # Shuffle to ensure we get a varied mix up to max_samples
    random.shuffle(all_candidates)

    matched_segments = []
    
    # Initialize counts based on what's already on disk
    root_counts = get_existing_counts(metadata_path, target_map)
    logging.info(f"Existing counts from metadata: {root_counts}")

    for segment in all_candidates:
        labels = segment['labels_ids']
        added = False
        
        for root, children in target_map.items():
            if children.intersection(labels):
                if root_counts[root] < max_samples:
                    root_counts[root] += 1
                    added = True
                    
        if added:
            matched_segments.append(segment)
            
        # Stop collecting candidates early if all categories hit their cap
        if all(count >= max_samples for count in root_counts.values()):
            logging.info("Candidate setup reached the MAX_SAMPLES_PER_CATEGORY limit.")
            break

    return matched_segments

def append_to_metadata(metadata_path, new_entry):
    """
    Appends a new entry to the metadata CSV, creating it if needed.
    """
    file_exists = metadata_path.exists()
    
    with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'labels_ids', 'labels_names', 'path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(new_entry)

def download_audio(segment, output_dir, id_to_item, metadata_path, target_os):
    """
    Downloads via yt-dlp and formats via ffmpeg.
    Returns True on success, False if skipped/failed.
    """
    ytid = segment['ytid']
    start_sec = segment['start_seconds']
    
    # Limit duration if necessary
    end_sec = min(segment['end_seconds'], start_sec + MAX_DURATION)
    
    base_filename = f"{ytid}_{start_sec}"
    wav_filename = f"{base_filename}.wav"
    output_path = output_dir / wav_filename
    
    # Skip if we already downloaded this exact file
    if output_path.exists():
        logging.info(f"[RESUME] '{wav_filename}' already exists. Skipping.")
        return False

    url = f"https://www.youtube.com/watch?v={ytid}"
    
    # Let yt-dlp determine the temp extension naturally, we enforce WAV via post-processor
    yt_output_template = str(output_dir / f"{base_filename}.%(ext)s")
    
    yt_dlp_bin = "yt-dlp.exe" if target_os == "windows" else "yt-dlp"
    
    cmd = [
        yt_dlp_bin,
        "-x",  # Equivalent to --extract-audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        "--postprocessor-args", f"ffmpeg:-ar {SAMPLE_RATE} -ac {CHANNELS}",
        "-o", yt_output_template,
    ]
    
    if USE_COOKIES:
        cmd.extend(["--cookies-from-browser", BROWSER])
        
    cmd.append(url)
    
    try:
        logging.info(f"Downloading [{base_filename}] | {min(end_sec - start_sec, MAX_DURATION)}s")
        # Directing terminal output to devnull so logs stay clean
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            # Slicing the error string ensures it doesn't flood the terminal
            logging.error(f"Download failed for video {ytid}. {result.stderr.strip()[:150]}")
            return False
            
        if not output_path.exists():
            logging.error(f"Couldn't locate the final output. The video {ytid} might be gone.")
            return False
            
        labels_names = [id_to_item[lbl]['name'] for lbl in segment['labels_ids'] if lbl in id_to_item]
        
        metadata_entry = {
            'filename': wav_filename,
            'labels_ids': ";".join(segment['labels_ids']),
            'labels_names': ";".join(labels_names),
            'path': str(output_path.absolute())
        }
        
        append_to_metadata(metadata_path, metadata_entry)
        return True
        
    except FileNotFoundError:
        logging.critical("yt-dlp or ffmpeg not found in PATH. Make sure they are installed!")
        raise
    except Exception as e:
        logging.error(f"Unexpected error on {ytid}: {str(e)}")
        return False

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    if not TARGET_CATEGORIES:
        logging.error("No target categories provided. Stopping.")
        return

    target_os = HOST_OS.lower()
    if target_os not in ["windows", "linux"]:
        logging.error("Invalid HOST_OS. Choose either 'windows' or 'linux'.")
        return
        
    logging.info(f"Configured Target OS: {target_os}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path("metadata.csv")
    
    logging.info(f"Saving output to: {output_dir.absolute()}")
    logging.info("Loading ontology map...")
    
    target_map, id_to_item = get_target_mappings(ONTOLOGY_PATH, TARGET_CATEGORIES)
    avoid_map, _ = get_target_mappings(ONTOLOGY_PATH, AVOID_CATEGORIES)
    
    if not target_map:
        logging.warning("Done. (Category config is empty or mismatched)")
        return
        
    resolved_count_sum = sum(len(v) for v in target_map.values())
    logging.info(f"Found {len(target_map)} target root categories mapping to {resolved_count_sum} specific tags.")
    if avoid_map:
        logging.info(f"Found {len(avoid_map)} blacklisted root categories, tracking {sum(len(v) for v in avoid_map.values())} specific tags to dodge.")
    
    logging.info("Parsing Unbalanced Train Dataset...")
    segments_to_process = parse_csv_for_targets(CSV_PATH, target_map, avoid_map, MAX_SAMPLES_PER_CATEGORY, MIN_DURATION, metadata_path)
    
    logging.info(f"Ready to process. Filtered candidates needed to top-up the batch: {len(segments_to_process)}")
    
    downloaded_count = 0

    for segment in segments_to_process:
        if downloaded_count >= BATCH_SIZE:
            logging.info(f"--> Reached batch limit [{BATCH_SIZE}]. Taking a break (re-run to grab more).")
            break
            
        is_success = download_audio(segment, output_dir, id_to_item, metadata_path, target_os)
        
        if is_success:
            downloaded_count += 1
            delay = random.uniform(SLEEP_INTERVAL[0], SLEEP_INTERVAL[1])
            logging.info(f"    --> Download complete ({downloaded_count}/{BATCH_SIZE}). Cooldown: {delay:.2f}s")
            time.sleep(delay)

    logging.info(f"All done! Successfully grabbed {downloaded_count} new samples this session.")

if __name__ == "__main__":
    main()
