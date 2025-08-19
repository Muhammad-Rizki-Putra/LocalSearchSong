import os
import glob
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import sqlite3
import argparse

# --- Database Constants ---
DB_PATH = 'fingerprints.db'

def init_db(db_path=DB_PATH):
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash INTEGER NOT NULL,
            song_id INTEGER NOT NULL,
            offset_time REAL NOT NULL,
            FOREIGN KEY (song_id) REFERENCES songs(id)
        )
    """)

    # THIS IS THE MOST IMPORTANT PART FOR RECOGNITION SPEED!
    # An index on the hash column allows for near-instant lookups.
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)")
    
    conn.commit()
    return conn, cur

def fingerprint_song(file_path):
    """
    Generates a landmark-based fingerprint for a single audio file.
    
    This function is mostly the same as your original, but it returns a
    list of (hash, timestamp) tuples instead of a dictionary, which is
    easier to process for database insertion.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        list: A list of tuples, where each tuple is (hash, anchor_time).
              Returns an empty list if an error occurs.
    """
    try:

        # loading song  
        y, sr = librosa.load(file_path)

        # fourier transformation
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        neighborhood_size = 15
        local_max = maximum_filter(S_db, footprint=np.ones((neighborhood_size, neighborhood_size)), mode='constant')
        detected_peaks = (S_db == local_max)
        amplitude_threshold = -50.0
        peaks = np.where((detected_peaks) & (S_db > amplitude_threshold))
        
        if not peaks[0].any():
            return []

        n_fft = (D.shape[0] - 1) * 2
        peak_freqs_at_peaks = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[peaks[0]]
        peak_times = librosa.frames_to_time(frames=peaks[1], sr=sr, n_fft=n_fft)
        peaks_list = list(zip(peak_times, peak_freqs_at_peaks))
        sorted_peaks = sorted(peaks_list, key=lambda p: p[0])

        fingerprints = []
        TARGET_ZONE_START_TIME = 0.1
        TARGET_ZONE_TIME_DURATION = 0.8
        TARGET_ZONE_FREQ_WIDTH = 200

        for i, anchor_peak in enumerate(sorted_peaks):
            anchor_time, anchor_freq = anchor_peak
            t_min = anchor_time + TARGET_ZONE_START_TIME
            t_max = t_min + TARGET_ZONE_TIME_DURATION
            f_min = anchor_freq - TARGET_ZONE_FREQ_WIDTH
            f_max = anchor_freq + TARGET_ZONE_FREQ_WIDTH
            
            for j in range(i + 1, len(sorted_peaks)):
                target_peak = sorted_peaks[j]
                target_time, target_freq = target_peak
                if target_time > t_max:
                    break
                if t_min <= target_time <= t_max and f_min <= target_freq <= f_max:
                    time_delta = target_time - anchor_time
                    h = hash((anchor_freq, target_freq, time_delta))
                    fingerprints.append((h, anchor_time))
                    
        return fingerprints

    except Exception as e:
        print(f"Could not process {file_path}. Error: {e}")
        return []

def add_song_to_db(conn, cur, file_path):
    """Processes a single song and adds its fingerprints to the database."""
    song_name = os.path.basename(file_path).replace('.mp3', '')
    
    cur.execute("SELECT id FROM songs WHERE name = ?", (song_name,))
    result = cur.fetchone()
    if result:
        print(f"'{song_name}' already exists in the database. Skipping.")
        return

    print(f"Processing: {song_name}...")
    
    # 1. Add song to the `songs` table to get a unique ID
    cur.execute("INSERT INTO songs (name) VALUES (?)", (song_name,))
    song_id = cur.lastrowid
    
    # 2. Generate the fingerprints for the song
    fingerprints = fingerprint_song(file_path)
    
    if not fingerprints:
        print(f"No fingerprints generated for {song_name}. Nothing to add.")
        # We can roll back the song insertion if no fingerprints were found
        conn.rollback()
        return
        
    # 3. Prepare fingerprints for bulk insertion
    data_to_insert = []
    for h, offset in fingerprints:
        data_to_insert.append((h, song_id, offset))
        
    # 4. Insert all fingerprints in a single, efficient transaction
    cur.executemany(
        "INSERT INTO fingerprints (hash, song_id, offset_time) VALUES (?, ?, ?)",
        data_to_insert
    )
    conn.commit()
    print(f"✅ Added {len(data_to_insert)} fingerprints for '{song_name}' to the database.")

def add_folder_to_db(conn, cur, folder_path):
    """Finds all .mp3 files in a folder and adds them to the database."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return
        
    print(f"\nScanning folder: {folder_path}")
    file_paths = glob.glob(os.path.join(folder_path, '*.mp3'))
    
    if not file_paths:
        print("No .mp3 files found in this folder.")
        return

    for file_path in file_paths:
        add_song_to_db(conn, cur, file_path)

# --- Main Execution Section ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add songs to the fingerprint database.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--song", type=str, help="Path to a single song file to add.")
    group.add_argument("--folder", type=str, help="Path to a folder of songs to add.")
    
    args = parser.parse_args()
    
    conn, cur = init_db()
    print("Database initialized.")
    
    if args.song:
        add_song_to_db(conn, cur, args.song)
    elif args.folder:
        add_folder_to_db(conn, cur, args.folder)
        
    print("\n--- Process Complete! ---")
    conn.close()