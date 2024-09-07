import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import os
from tqdm import tqdm

MODEL_PATH = 'song_matching_model.joblib'
SCALER_PATH = 'feature_scaler.joblib'
DATABASE_INFO_PATH = 'database_info.joblib'

def extract_features(audio_file):
    """Extract relevant features from the audio file."""
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
    """Create a database of songs from a folder of audio files."""
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
    """Train a machine learning model for song comparison."""
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print("Training Nearest Neighbors model...")
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(scaled_features)
    
    return model, scaler

def find_matching_songs(input_file, model, scaler, database_features, file_names, n_matches=5):
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
    
    return [m[0] for m in matches], [m[1] for m in matches]

def build_database(music_dir):
    """Build and save the song database, model, and scaler."""
    database_features, file_names = create_song_database(music_dir)
    model, scaler = train_model(database_features)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump((database_features, file_names), DATABASE_INFO_PATH)

def load_model_and_database():
    """Load the existing model, scaler, and database information."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    database_features, file_names = joblib.load(DATABASE_INFO_PATH)
    return model, scaler, database_features, file_names

def get_matching_songs(input_song_path, music_dir, n_matches=5):
    """Main function to get matching songs, building the database if necessary."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(DATABASE_INFO_PATH):
        build_database(music_dir)
    
    model, scaler, database_features, file_names = load_model_and_database()
    matching_songs, distances = find_matching_songs(input_song_path, model, scaler, database_features, file_names, n_matches)
    
    return matching_songs, distances

if __name__ == "__main__":
    # This block is for testing purposes only
    music_dir = "music/"
    input_song = "music/song.wav"
    matches = get_matching_songs(input_song, music_dir)
    print("Matching songs:", matches)