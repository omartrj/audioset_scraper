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
# CONFIGURAZIONE INIZIALE
# ==============================================================================

# Categorie target: definisci un array di stringhe (id o nomi)
TARGET_CATEGORIES = ["Human voice", "Human group actions", "Motor vehicle (road)"]

# Categorie da EVITARE: scarta video che contengono questi suoni (o i loro figli)
AVOID_CATEGORIES = ["Emergency vehicle"]

# Sistema operativo dell'host: "windows" o "linux"
HOST_OS = "linux"  # <--- CAMBIA QUESTO VALORE SE NECESSARIO

# Limite intero per evitare che una classe domini il dataset (es: max 100 sample per root category)
MAX_SAMPLES_PER_CATEGORY = 2  # <- Abbassato a 2 per test rapido

# Directory di destinazione unica 
OUTPUT_DIR = "downloaded_audio"

# Audio Settings
SAMPLE_RATE = 48000
CHANNELS = 1              # 1 = Mono
MAX_DURATION = 10         # Durata massima in secondi

# Browser & download Settings
USE_COOKIES = False       # Setta a True in caso di problemi di restrizione età/bot (richiede browser cookies)
BROWSER = "chrome"        # Scegli uno tra: chrome, firefox, edge, safari, opera, brave
SLEEP_INTERVAL = [2, 5]   # Min e Max delay tra un download e l'altro in secondi
BATCH_SIZE = 5           # <- Abbassato a 5 per test rapido

# Percorsi ai meta-dati necessari di AudioSet
ONTOLOGY_PATH = "ontology.json"
CSV_PATH = "unbalanced_train_segments.csv"

# ==============================================================================
# CONFIGURAZIONE LOGGING
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================================================================
# FUNZIONI PRINCIPALI
# ==============================================================================

def get_target_mappings(ontology_path, target_categories):
    """
    Risolve ricorsivamente i child_ids delle categorie in `target_categories`.
    Ritorna un dizionario `{root_id: set_di_sub_ids_inclusa_root}` per bilanciare correttamente il dataset.
    """
    try:
        with open(ontology_path, 'r', encoding='utf-8') as f:
            ontology = json.load(f)
    except FileNotFoundError:
        logging.error(f"File {ontology_path} non trovato.")
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
            logging.warning(f"Categoria ignorata (non trovata nell'ontologia): {cat}")

    return target_map, id_to_item

def parse_csv_for_targets(csv_path, target_map, avoid_map, max_samples):
    """
    Scansiona il file CSV, identifica i video pertinenti e ritorna una lista 
    che rispetta `max_samples` massimo per ogni categoria (root).
    Scarta tutti i video che possiedono tag presenti in `avoid_map`.
    """
    if not Path(csv_path).exists():
        logging.error(f"File CSV non trovato: {csv_path}")
        return []

    # Flatten di tutti gli id validi per check rapido
    all_valid_ids = set()
    for children in target_map.values():
        all_valid_ids.update(children)
        
    # Flatten degli id da evitare
    all_avoid_ids = set()
    if avoid_map:
        for children in avoid_map.values():
            all_avoid_ids.update(children)

    all_candidates = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # skipinitialspace per ignorare gli spazi dopo ogni singola virgola
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                # Ignora le righe di commento o vuote
                if not row or row[0].startswith('#'):
                    continue
                
                # I metadata di audioset CSV si presentano così: yt_id, start_seconds, end_seconds, labels...
                ytid = row[0]
                start_sec = float(row[1])
                end_sec = float(row[2])
                
                # Uniamo tutto ciò che ricade da row[3] in avanti (visto che audioset raggruppa multipli labels splitagati per virgola)
                labels_str = ",".join(row[3:]).replace('"', '').strip()
                labels_list = [l.strip() for l in labels_str.split(',')]
                
                # Controllo super rapido con i valid_ids e avoid_ids
                if all_valid_ids.intersection(labels_list):
                    if not all_avoid_ids.intersection(labels_list):  # Se NON contiene etichette da scartare
                        all_candidates.append({
                            'ytid': ytid,
                            'start_seconds': start_sec,
                            'end_seconds': end_sec,
                            'labels_ids': labels_list
                        })
    except Exception as e:
        logging.error(f"Errore durante la lettura del CSV: {e}")
        return []

    # Mescoliamo i campioni per dare varietà (poiché limiteremo via "max_samples"!)
    random.shuffle(all_candidates)

    matched_segments = []
    root_counts = {r: 0 for r in target_map.keys()}

    for segment in all_candidates:
        labels = segment['labels_ids']
        added = False
        
        # Ci assicuriamo di non sforare il limite per l'assegnazione
        for root, children in target_map.items():
            if children.intersection(labels):
                if root_counts[root] < max_samples:
                    root_counts[root] += 1
                    added = True
                    
        if added:
            matched_segments.append(segment)
            
        # Break condizionale se tutti i root id hanno soddisfatto la capienza
        if all(count >= max_samples for count in root_counts.values()):
            logging.info("Raggiunto il MAX_SAMPLES_PER_CATEGORY per la fase di setup candidati.")
            break

    return matched_segments

def append_to_metadata(metadata_path, new_entry):
    """
    Genera od aggiorna (append) il metadata.csv.
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
    Usa yt-dlp per scaricare la traccia audio filtrata e converte via ffmpeg 
    rispettando la direttiva durata, rate e mono.
    Ritorna True in caso di download riuscito, bloccato/fallito=False.
    """
    ytid = segment['ytid']
    start_sec = segment['start_seconds']
    
    # Processiamo max durata:
    end_sec = min(segment['end_seconds'], start_sec + MAX_DURATION)
    
    base_filename = f"{ytid}_{start_sec}"
    wav_filename = f"{base_filename}.wav"
    output_path = output_dir / wav_filename
    
    # --- Check Resume ---
    if output_path.exists():
        logging.info(f"[RESUME] File '{wav_filename}' già esistente. Skkippato.")
        # Ritorniamo falso perché non lo contabilizziamo nei download della current BATCH
        return False

    url = f"https://www.youtube.com/watch?v={ytid}"
    
    # Template path yt-dlp non accetta l'estensione rigida se noi le passiamo flag --audio-format
    # Per cui usiamo .%(ext)s ed avvantaggiamo il fatto che estrarrà wav puro con la postprocessor flag.
    # Convertiamo a stringa per garantire path cross-platform in compatibilità subprocess
    yt_output_template = str(output_dir / f"{base_filename}.%(ext)s")
    
    yt_dlp_bin = "yt-dlp.exe" if target_os == "windows" else "yt-dlp"
    
    cmd = [
        yt_dlp_bin,
        "-x",  # equivale a --extract-audio
        "--audio-format", "wav",
        "--audio-quality", "0",
        # Taglia la sezione corretta
        "--download-sections", f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        # Post-processor argument for ffmpeg (Resampling/Mono/ecc)
        "--postprocessor-args", f"ffmpeg:-ar {SAMPLE_RATE} -ac {CHANNELS}",
        "-o", yt_output_template,
    ]
    
    if USE_COOKIES:
        cmd.extend(["--cookies-from-browser", BROWSER])
        
    # Appendiamo l'URL in fondo alla lista argomenti
    cmd.append(url)
    
    try:
        logging.info(f"Downloading [{base_filename}] | {min(end_sec - start_sec, MAX_DURATION)}s")
        # Redirezionamento dei log di base yt-dlp e passandoli al nostro devnull (salvo errori visibili in stderr)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            logging.error(f"Errore al download del video: {ytid}. {result.stderr.strip()[:150]}") # Troncato x leggibilità
            return False
            
        if not output_path.exists():
            logging.error(f"Impossibile posizionare output, forse il video base non esiste più per ID: {ytid}")
            return False
            
        # Generiamo le descrizioni dei labels lette
        labels_names = [id_to_item[lbl]['name'] for lbl in segment['labels_ids'] if lbl in id_to_item]
        
        metadata_entry = {
            'filename': wav_filename,
            'labels_ids': ";".join(segment['labels_ids']),   # delimiti interno a punto e virgola
            'labels_names': ";".join(labels_names),
            'path': str(output_path.absolute()) # Compatibile robusto x Windows e Linux
        }
        
        append_to_metadata(metadata_path, metadata_entry)
        return True
        
    except FileNotFoundError:
        logging.critical("yt-dlp o ffmpeg NON trovato nel PATH di sistema. Assicurati che siano installati correttamente!")
        # Rilanciamo o forziamo exit visto che non è recuperabile questo state
        raise
    except Exception as e:
        logging.error(f"Riepilogo Errore su yt_id {ytid}: {str(e)}")
        return False

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    if not TARGET_CATEGORIES:
        logging.error("Nessuna categoria target presente in TARGET_CATEGORIES. Script arrestato.")
        return

    target_os = HOST_OS.lower()
    if target_os not in ["windows", "linux"]:
        logging.error("HOST_OS non valido. Scegli tra 'windows' o 'linux'.")
        return
        
    logging.info(f"OS Target configurato (codice HOST_OS): {target_os}")

    # Usando pathlib per path indipendenti dalle sbarre OS-specifiche (Windows / linux)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    
    logging.info(f"Target di salvataggio directory: {output_dir.absolute()}")
    logging.info("Caricamento configurazioni categorie da Ontology...")
    
    target_map, id_to_item = get_target_mappings(ONTOLOGY_PATH, TARGET_CATEGORIES)
    avoid_map, _ = get_target_mappings(ONTOLOGY_PATH, AVOID_CATEGORIES)
    
    if not target_map:
        logging.warning("Finito. (Configurazione Categorie assente o non corrispondente)")
        return
        
    # Count of valid resolved targets
    resolved_count_sum = sum(len(v) for v in target_map.values())
    logging.info(f"Trovate {len(target_map)} categorie Root che matchano un totale inclusivo di {resolved_count_sum} tag specifici.")
    if avoid_map:
        logging.info(f"Trovate {len(avoid_map)} categorie Root DA SCARTARE, corrispondenti a {sum(len(v) for v in avoid_map.values())} tag specifici.")
    
    logging.info("Parsando Unbalanced Train Dataset...")
    segments_to_process = parse_csv_for_targets(CSV_PATH, target_map, avoid_map, MAX_SAMPLES_PER_CATEGORY)
    
    logging.info(f"Pronto per l'elaborazione. Candidati pre-selezionati per coprire limite batch: {len(segments_to_process)}")
    
    downloaded_count = 0

    for segment in segments_to_process:
        if downloaded_count >= BATCH_SIZE:
            logging.info(f"--> Raggiunto limite BATCH_SIZE [{BATCH_SIZE}]. Esecuzione terminata (riavvia se occorre scaricare altri segmenti).")
            break
            
        is_success = download_audio(segment, output_dir, id_to_item, metadata_path, target_os)
        
        if is_success:
            downloaded_count += 1
            delay = random.uniform(SLEEP_INTERVAL[0], SLEEP_INTERVAL[1])
            logging.info(f"    --> Download effettuato ({downloaded_count}/{BATCH_SIZE}). Delay applicato: {delay:.2f}s per emulare traffico umano.")
            time.sleep(delay)

    logging.info(f"Lavoro concluso! Riusciti nuovi scaricamenti in questa sessione: {downloaded_count}.")

if __name__ == "__main__":
    main()
