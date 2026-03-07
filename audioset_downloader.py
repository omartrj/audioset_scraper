import json
import csv
import time
import random
import subprocess
import logging
import argparse
from pathlib import Path

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

def get_existing_downloads(metadata_path):
    """
    Reads metadata.csv (if it exists) and returns a set of already-downloaded
    filenames (without extension) so we can skip them on resume.
    """
    if not metadata_path.exists():
        return set()
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # filename is stored as "ytid_start.wav", strip the extension for key matching
            return {Path(row['filename']).stem for row in reader}
    except Exception as e:
        logging.error(f"Failed reading metadata.csv: {e}")
        return set()

def parse_csv_for_targets(csv_path, target_map, avoid_map, max_samples, min_duration, metadata_path):
    """
    Scans the CSV, drops short clips, drops clips with unwanted labels,
    and returns up to max_samples candidates (minus those already downloaded).
    """
    if not Path(csv_path).exists():
        logging.error(f"CSV file not found: {csv_path}")
        return []

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
                labels_set = set(labels_list)
                
                # AND logic: the clip must contain at least one label from EVERY target category
                if all(children.intersection(labels_set) for children in target_map.values()):
                    if not all_avoid_ids.intersection(labels_set):
                        all_candidates.append({
                            'ytid': ytid,
                            'start_seconds': start_sec,
                            'end_seconds': end_sec,
                            'labels_ids': labels_list
                        })
    except Exception as e:
        logging.error(f"Error parsing the CSV: {e}")
        return []

    # Shuffle to ensure we get a varied mix
    random.shuffle(all_candidates)

    # Filter out segments already downloaded (prevents duplicates on resume)
    already_downloaded = get_existing_downloads(metadata_path)
    logging.info(f"Already downloaded: {len(already_downloaded)} samples")
    all_candidates = [
        s for s in all_candidates
        if f"{s['ytid']}_{s['start_seconds']}" not in already_downloaded
    ]

    remaining = max_samples - len(already_downloaded)
    if remaining <= 0:
        logging.info("MAX_SAMPLES already reached. Nothing to download.")
        return []

    # Return ALL valid candidates so failures don't reduce the final count
    return all_candidates

def append_to_metadata(metadata_path, new_entry):
    """
    Appends a new entry to the metadata CSV, creating it if needed.
    """
    file_exists = metadata_path.exists()
    
    with open(metadata_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'labels_ids', 'labels_names']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(new_entry)

def download_audio(segment, output_dir, id_to_item, metadata_path, cfg):
    """
    Downloads via yt-dlp and formats via ffmpeg.
    Returns True on success, False if skipped/failed.
    """
    ytid = segment['ytid']
    start_sec = segment['start_seconds']

    max_duration = cfg["max_duration"]
    sample_rate = cfg["sample_rate"]
    channels = cfg["channels"]
    use_cookies = cfg["use_cookies"]
    browser_name = cfg["browser_name"]
    target_os = cfg["host"]
    
    # Limit duration if necessary
    end_sec = min(segment['end_seconds'], start_sec + max_duration)
    
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
        "--no-warnings",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac {channels}",
        "-o", yt_output_template,
    ]
    
    if use_cookies:
        cmd.extend(["--cookies-from-browser", browser_name])
        
    cmd.append(url)
    
    try:
        logging.info(f"Downloading [{base_filename}] | {min(end_sec - start_sec, max_duration)}s")
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
            'filename': f"audio/{wav_filename}",
            'labels_ids': ";".join(segment['labels_ids']),
            'labels_names': ";".join(labels_names)
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
    parser = argparse.ArgumentParser(description="AudioSet Downloader")
    parser.add_argument("--config", required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dl = raw.get("download", {})
    audio_cfg = raw.get("audio", {})
    browser_cfg = raw.get("browser", {})

    cfg = {
        # [download]
        "targets":      dl["targets"],
        "avoid":        dl.get("avoid", []),
        "host":         dl.get("host", "linux").lower(),
        "max_samples":  dl["max_samples"],
        "output_dir":   dl["output_dir"],
        "metadata":     dl["metadata"],
        "batch_size":   dl.get("batch_size", 500),
        "csv_path":     dl["csv_path"],
        "ontology_path": dl.get("ontology_path", "ontology.json"),
        # [audio]
        "sample_rate":  audio_cfg.get("sample_rate", 48000),
        "channels":     audio_cfg.get("channels", 1),
        "min_duration": audio_cfg.get("min_duration", 8),
        "max_duration": audio_cfg.get("max_duration", 10),
        # [browser]
        "use_cookies":  browser_cfg.get("use_cookies", False),
        "browser_name": browser_cfg.get("browser", "firefox"),
        "sleep_interval": browser_cfg.get("sleep_interval", [1, 3]),
    }

    if not cfg["targets"]:
        logging.error("No target categories provided. Stopping.")
        return

    if cfg["host"] not in ["windows", "linux"]:
        logging.error("Invalid host in config. Choose either 'windows' or 'linux'.")
        return

    logging.info(f"Configured Target OS: {cfg['host']}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / cfg["metadata"]

    logging.info(f"Saving output to: {output_dir.absolute()}")
    logging.info("Loading ontology map...")

    target_map, id_to_item = get_target_mappings(cfg["ontology_path"], cfg["targets"])
    avoid_map, _ = get_target_mappings(cfg["ontology_path"], cfg["avoid"])

    if not target_map:
        logging.warning("Done. (Category config is empty or mismatched)")
        return

    resolved_count_sum = sum(len(v) for v in target_map.values())
    logging.info(f"Found {len(target_map)} target root categories mapping to {resolved_count_sum} specific tags.")
    if avoid_map:
        logging.info(f"Found {len(avoid_map)} blacklisted root categories, tracking {sum(len(v) for v in avoid_map.values())} specific tags to dodge.")

    logging.info(f"Parsing dataset: {cfg['csv_path']} ...")
    segments_to_process = parse_csv_for_targets(
        cfg["csv_path"], target_map, avoid_map,
        cfg["max_samples"], cfg["min_duration"], metadata_path
    )
    already_done = len(get_existing_downloads(metadata_path))

    logging.info(f"Ready to process. Filtered candidates available: {len(segments_to_process)}")

    downloaded_count = 0
    sleep_min, sleep_max = cfg["sleep_interval"][0], cfg["sleep_interval"][1]

    for segment in segments_to_process:
        total_so_far = already_done + downloaded_count
        if total_so_far >= cfg["max_samples"]:
            logging.info(f"--> Reached MAX_SAMPLES [{cfg['max_samples']}]. Done.")
            break
        if downloaded_count >= cfg["batch_size"]:
            logging.info(f"--> Reached batch limit [{cfg['batch_size']}]. Taking a break (re-run to grab more).")
            break

        is_success = download_audio(segment, audio_dir, id_to_item, metadata_path, cfg)

        if is_success:
            downloaded_count += 1
            delay = random.uniform(sleep_min, sleep_max)
            logging.info(f"    --> Download complete ({already_done + downloaded_count}/{cfg['max_samples']}). Cooldown: {delay:.2f}s")
            time.sleep(delay)

    logging.info(f"All done! Successfully grabbed {downloaded_count} new samples this session.")

if __name__ == "__main__":
    main()
