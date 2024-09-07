import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os
from tqdm import tqdm
import argparse

def extract_features(audio_file):
    """
    Extract relevant features from the audio file.
    """
    y, sr = librosa.load(audio_file)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    features = np.hstack([
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff),
        tempo,
        np.mean(mfcc, axis=1),
        np.mean(zcr)
    ])
    
    return features

def create_song_database(folder_path):
    """
    Create a database of songs from a folder of audio files.
    """
    features_list = []
    file_names = []
    
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.flac'))]
    
    for file in tqdm(audio_files, desc="Extracting features", unit="song"):
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        features_list.append(features)
        file_names.append(file)
    
    return np.array(features_list), file_names

def train_model(features):
    """
    Train a machine learning model for song comparison.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print("Training Nearest Neighbors model...")
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(scaled_features)
    
    return model, scaler

def find_matching_songs(input_file, model, scaler, database_features, file_names, n_matches=5):
    """
    Find the top N matching songs for the input file, excluding self-matches.
    """
    print("Extracting features from input song...")
    input_features = extract_features(input_file)
    scaled_input = scaler.transform(input_features.reshape(1, -1))
    
    print(f"Finding top {n_matches} nearest neighbors...")
    distances, indices = model.kneighbors(scaled_input, n_neighbors=n_matches + 1)
    
    input_file_name = os.path.basename(input_file)
    
    matches = []
    for i, index in enumerate(indices[0]):
        if file_names[index] != input_file_name:
            matches.append((file_names[index], 1 - distances[0][i]))
        
        if len(matches) == n_matches:
            break
    
    return matches

def main(build_database, input_song_path, n_matches):
    database_folder = "songs/"
    model_path = 'song_matching_model.joblib'
    scaler_path = 'feature_scaler.joblib'

    if build_database:
        print("Creating song database...")
        database_features, file_names = create_song_database(database_folder)
        
        model, scaler = train_model(database_features)
        
        print("Saving model and scaler...")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump((database_features, file_names), 'database_info.joblib')
    else:
        print("Loading existing model and database...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        database_features, file_names = joblib.load('database_info.joblib')

    matching_songs = find_matching_songs(input_song_path, model, scaler, database_features, file_names, n_matches)
    
    if matching_songs:
        print(f"Top {n_matches} matching songs:")
        for i, (song, score) in enumerate(matching_songs, 1):
            print(f"{i}. {song} (Similarity score: {score:.2f})")
    else:
        print("No matching songs found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHLong")
    parser.add_argument("--build-database", action="store_true", help="Build the song database")
    parser.add_argument("--input-song", required=True, help="Path to the input song file")
    parser.add_argument("--n-matches", type=int, default=5, help="Number of top matches to return")
    args = parser.parse_args()

    main(args.build_database, args.input_song, args.n_matches)