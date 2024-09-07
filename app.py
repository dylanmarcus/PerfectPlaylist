from flask import Flask, render_template, send_file
import os
from match_songs import get_matching_songs

app = Flask(__name__)

MUSIC_DIR = 'songs/'

def get_songs():
    return [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(('.mp3', '.wav'))]

@app.route('/')
def index():
    songs = get_songs()
    return render_template('index.html', songs=songs, view='library')

@app.route('/play/<path:song>')
def play_song(song):
    return send_file(os.path.join(MUSIC_DIR, song))

@app.route('/playlist/<path:song>')
def show_playlist(song):
    input_song_path = os.path.join(MUSIC_DIR, song)
    matching_songs, distances = get_matching_songs(input_song_path, MUSIC_DIR, n_matches=10)
    return render_template('index.html', songs=matching_songs, distances=distances, view='playlist', selected_song=song, zip=zip)

if __name__ == '__main__':
    app.run(debug=True)