# Audio Fingerprinting & Audio Search Engine  
---  

A repository dedicated to an audio search system that uses fingerprinting techniques to identify audio files stored in a database.  

## Features  

- **Fingerprint Generation**: Converts audio files into unique fingerprints (hashes) stored in a database.  
- **Audio Search**: Identifies songs from trimmed/partial audio samples by matching fingerprints against the database.  

## Pipeline  

### 1. Fingerprint Creation & Database Insertion  
1. **Audio Loading**: Load the audio file using `librosa`.  
2. **Spectrogram Conversion**: Convert the audio into a spectrogram using Short-Time Fourier Transform (STFT).  
3. **Peak Extraction**: Detect high-intensity peaks from the spectrogram to generate a constellation map.  
4. **Fingerprint Hashing**: Create unique fingerprints (hashes) from the constellation peaks.  
5. **Database Insertion**: Store the generated hashes in the database.  

### 2. Audio Search Process  
1. **Query Audio Processing**: Load and convert the input audio into fingerprints (same as Step 1).  
2. **Database Lookup**: Match the query fingerprints against the database.  
3. **Voting Mechanism**: Rank results based on the highest number of fingerprint matches.  

## Installation  

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt

## How to Run

1. run `pip install requirements.txt`
2. a. for adding a mp3 song to database `python MP3toDatabase.py --song 'asong.mp3'`, for folder which contain mp3 `python MP3toDatabase.py --folder 'mp3_folder'`
3. b. for searh audio `python SearchSong.py TrimmedSong.mp3`
