import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import sqlite3
import argparse
from collections import defaultdict

# --- Database Constants ---
DB_PATH = 'fingerprints.db'

def fingerprint_song(file_path):
    """
    Generates a landmark-based fingerprint for a single audio file.
    
    This function is identical to the one in the database creation script.
    It's needed here to process the query clip.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        list: A list of tuples, where each tuple is (hash, anchor_time).
              Returns an empty list if an error occurs.
    """

    try:
        y, sr = librosa.load(file_path, duration=20) # Load only a portion for recognition

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
        print(f"Could not process query file {file_path}. Error: {e}")
        return []

def recognize_from_db(query_path, db_path=DB_PATH):
    """
    Searches for a query audio clip in the SQLite fingerprint database.
    
    Args:
        query_path (str): Path to the short audio clip to identify.
        db_path (str): Path to the SQLite database file.
        
    Returns:
        tuple: (best_matching_song_name, confidence_score) or None if no match is found.
    """
    if not os.path.exists(db_path):
        print(f"Database file not found at '{db_path}'. Please create it first.")
        return None

    # 1. Fingerprint the query clip
    print(f"Fingerprinting query file: {query_path}...")
    query_fingerprints = fingerprint_song(query_path)
    
    if not query_fingerprints:
        print("Could not generate fingerprints for the query clip.")
        return None
        
    # Create a dictionary for quick lookup of query timestamps by hash
    query_hashes = {h: t for h, t in query_fingerprints}
    
    # 2. Query the database to find all hashes that match our query
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create placeholders for the IN clause
    placeholders = ', '.join('?' for _ in query_hashes)
    
    # This single query fetches all potential matches at once. It's very fast.
    sql_query = f"""
        SELECT f.hash, s.name, f.offset_time
        FROM fingerprints f
        JOIN songs s ON s.id = f.song_id
        WHERE f.hash IN ({placeholders})
    """
    
    db_matches = cur.execute(sql_query, list(query_hashes.keys())).fetchall()
    conn.close()
    
    if not db_matches:
        return None

    # 3. Process matches and "vote" for the best (song, offset) pair
    matches = defaultdict(int) # A dictionary to store vote counts
    
    print(f"Found {len(db_matches)} potential hash matches in the database.")
    
    for h, song_name, db_timestamp in db_matches:
        query_timestamp = query_hashes[h]
        offset = round(db_timestamp - query_timestamp, 2) # Round offset to reduce noise
        key = (song_name, offset)
        matches[key] += 1
        
    # 4. Find the best match with the highest number of votes
    if not matches:
        return None
        
    best_match = max(matches.items(), key=lambda item: item[1])
    (song_name, offset), score = best_match
    
    return (song_name, score)


# --- Main Execution Section ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize a song from an audio clip.")
    parser.add_argument("query_file", type=str, help="Path to the audio clip you want to identify.")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to the fingerprint database file.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found at '{args.query_file}'")
    else:
        result = recognize_from_db(args.query_file, args.db)
        
        print("\n--- Recognition Result ---")
        if result:
            song, score = result
            print(f"✅ Match Found: '{song}'")
            print(f"   Confidence Score: {score} (number of matching hash pairs)")
        else:
            print("❌ No match found in the database.")