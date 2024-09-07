let currentSound;
let currentSong;

function playSong(song) {
    if (currentSound) {
        currentSound.unload();
    }

    currentSong = song;
    document.getElementById('nowPlaying').innerText = `Now Playing: ${song}`;

    currentSound = new Howl({
        src: [`/play/${encodeURIComponent(song)}`],
        format: ['mp3', 'wav', 'ogg'],
        onplay: function() {
            requestAnimationFrame(updateProgress);
        },
        onend: function() {
            document.querySelector('.progress-bar').style.width = '0%';
        }
    });

    currentSound.play();
}

function togglePlay() {
    if (currentSound) {
        if (currentSound.playing()) {
            currentSound.pause();
        } else {
            currentSound.play();
        }
    }
}

function updateProgress() {
    if (currentSound && currentSound.playing()) {
        const progress = (currentSound.seek() / currentSound.duration()) * 100;
        document.querySelector('.progress-bar').style.width = `${progress}%`;
        requestAnimationFrame(updateProgress);
    }
}