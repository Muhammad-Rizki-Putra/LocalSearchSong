import os
import librosa
import numpy as np
from scipy.ndimage import maximum_filter
import sqlite3
import argparse
from collections import defaultdict
import tempfile

# --- Database Constants ---
DB_PATH = 'fingerprints.db'

def fingerprint_song(file_path, start_time=0, duration=None):
    """
    Generates a landmark-based fingerprint for an audio file or segment.
    
    Args:
        file_path (str): Path to the audio file.
        start_time (float): Start time in seconds for the segment to process.
        duration (float): Duration in seconds of the segment to process.
        
    Returns:
        list: A list of tuples, where each tuple is (hash, anchor_time).
              Returns an empty list if an error occurs.
    """

    try:
        # Load only a specific segment if duration is specified
        if duration is not None:
            y, sr = librosa.load(file_path, offset=start_time, duration=duration)
        else:
            y, sr = librosa.load(file_path)

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

def recognize_from_db(query_path, db_path=DB_PATH, start_time=0, duration=None):
    """
    Searches for a query audio clip in the SQLite fingerprint database.
    
    Args:
        query_path (str): Path to the audio clip to identify.
        db_path (str): Path to the SQLite database file.
        start_time (float): Start time in seconds for the segment to process.
        duration (float): Duration in seconds of the segment to process.
        
    Returns:
        tuple: (best_matching_song_name, confidence_score, offset_time) or None if no match is found.
    """
    if not os.path.exists(db_path):
        print(f"Database file not found at '{db_path}'. Please create it first.")
        return None

    print(f"Fingerprinting query file segment: {query_path} (from {start_time}s to {start_time + duration if duration else 'end'}s)...")
    query_fingerprints = fingerprint_song(query_path, start_time, duration)
    
    if not query_fingerprints:
        print("Could not generate fingerprints for the query clip segment.")
        return None
        
    query_hashes = {h: t for h, t in query_fingerprints}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    placeholders = ', '.join('?' for _ in query_hashes)
    
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

    matches = defaultdict(int) 
    offset_details = {}
    
    for h, song_name, db_timestamp in db_matches:
        query_timestamp = query_hashes[h]
        offset = round(db_timestamp - query_timestamp, 2) # Round offset to reduce noise
        key = (song_name, offset)
        matches[key] += 1
        
    if not matches:
        return None
        
    best_match = max(matches.items(), key=lambda item: item[1])
    (song_name, offset), score = best_match
    
    return (song_name, score, offset)

def format_time_position(seconds):
    """Convert seconds to minutes:seconds format"""
    minutes = int(seconds // 60)
    seconds_remaining = int(seconds % 60)
    return f"{minutes}:{seconds_remaining:02d}"

def detect_multiple_songs(file_path, db_path=DB_PATH, segment_duration=15, min_confidence=1, overlap=5):
    """
    Detects multiple songs in a single audio file by processing it in segments.
    
    Args:
        file_path (str): Path to the audio file to analyze.
        db_path (str): Path to the fingerprint database.
        segment_duration (int): Duration of each segment to analyze in seconds.
        min_confidence (int): Minimum number of matching hashes to consider a match valid.
        overlap (int): Overlap between segments in seconds to avoid missing transitions.
        
    Returns:
        list: List of detected songs with their start times and confidence scores.
    """
    # Get the total duration of the audio file
    y, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Analyzing file of duration {total_duration:.2f} seconds...")
    
    detected_songs = []
    current_pos = 0
    last_detected_song = None
    
    while current_pos < total_duration:
        # Process the current segment
        result = recognize_from_db(file_path, db_path, current_pos, segment_duration)
        
        if result:
            song, score, offset = result
            
            # Only consider matches with sufficient confidence
            if score >= min_confidence:
                # Calculate the actual position in the file where this song starts
                song_start_in_file = current_pos + offset
                
                # If this is the same song as the previous detection, just update the end time
                if last_detected_song and last_detected_song["song"] == song:
                    last_detected_song["end_time"] = current_pos + segment_duration
                    last_detected_song["confidence"] = max(last_detected_song["confidence"], score)
                else:
                    # New song detected
                    song_info = {
                        "song": song,
                        "start_time": song_start_in_file,
                        "end_time": current_pos + segment_duration,
                        "confidence": score
                    }
                    detected_songs.append(song_info)
                    last_detected_song = song_info
                    
                    print(f"Detected '{song}' starting at {format_time_position(song_start_in_file)} (confidence: {score})")
        
        # Move to the next segment with overlap
        current_pos += segment_duration - overlap
    
    # Merge adjacent segments of the same song
    merged_detections = []
    for song_info in detected_songs:
        if merged_detections and merged_detections[-1]["song"] == song_info["song"]:
            # Extend the previous detection
            merged_detections[-1]["end_time"] = song_info["end_time"]
            merged_detections[-1]["confidence"] = max(merged_detections[-1]["confidence"], song_info["confidence"])
        else:
            # Add as new detection
            merged_detections.append(song_info)
    
    return merged_detections

# --- Main Execution Section ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize songs from an audio file, potentially containing multiple songs.")
    parser.add_argument("query_file", type=str, help="Path to the audio file you want to analyze.")
    parser.add_argument("--db", type=str, default=DB_PATH, help="Path to the fingerprint database file.")
    parser.add_argument("--segment", type=int, default=15, help="Duration of each analysis segment in seconds.")
    parser.add_argument("--min-confidence", type=int, default=10, help="Minimum number of matching hashes to consider a valid match.")
    parser.add_argument("--overlap", type=int, default=5, help="Overlap between segments in seconds.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found at '{args.query_file}'")
    else:
        results = detect_multiple_songs(
            args.query_file, 
            args.db, 
            args.segment, 
            args.min_confidence, 
            args.overlap
        )
        
        print("\n--- Multiple Song Detection Results ---")
        if results:
            for i, result in enumerate(results, 1):
                start_pos = format_time_position(result["start_time"])
                end_pos = format_time_position(result["end_time"])
                print(f"{i}. '{result['song']}'")
                print(f"   Time: {start_pos} - {end_pos}")
                print(f"   Confidence: {result['confidence']}")
        else:
            print("❌ No songs detected in the file.")