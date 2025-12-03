import librosa
import numpy as np
import soundfile as sf

class ExtractAudio:
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128, max_len=200):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_len = max_len
        self.sr = None   # sample rate will be set when loading audio

    def extract_spectrogram(self, file_path):
        # Load audio
        y, self.sr = librosa.load(file_path, sr=None)

        # Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Pad or truncate
        if S_db.shape[1] < self.max_len:
            pad_width = self.max_len - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_db = S_db[:, :self.max_len]

        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return S_norm
    
    def extract_spectrogram_patch(self, y, sr, onset_frame, n_fft=2048, hop_length=512, n_mels=128, win_frames=200):
        """Extract a small mel-spectrogram patch around an onset."""
        start = max(0, onset_frame - win_frames//2)
        end = onset_frame + win_frames//2

        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Clip to patch
        if end > S_db.shape[1]:
            pad_width = end - S_db.shape[1]
            S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode="constant")

        patch = S_db[:, start:end]

        if patch.shape[1] != win_frames:
            patch = librosa.util.fix_length(patch, size=win_frames, axis=1)

        # Normalize 0–1
        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)

        return patch.reshape(-1)   # flatten for NN input

    def spectrogram_to_audio(self, S_db, out_file=None):
        # Convert dB back to power
        S_power = librosa.db_to_power(S_db)

        # Convert Mel → linear-frequency STFT
        S_inv = librosa.feature.inverse.mel_to_stft(
            S_power, sr=self.sr, n_fft=self.n_fft
        )

        # Reconstruct waveform with Griffin–Lim
        y_inv = librosa.griffinlim(
            S_inv, hop_length=self.hop_length, n_fft=self.n_fft
        )

        # Save file if requested
        if out_file:
            sf.write(out_file, y_inv, self.sr)
            print(f"Reconstructed audio saved to {out_file}")

        return y_inv
    
    def verify_round_trip(self, file_path, out_file="roundtrip.wav"):
        """Test audio -> spectrogram -> audio round trip."""
        # Step 1: Extract spectrogram
        S_db = self.extract_spectrogram(file_path)

        # Step 2: Reconstruct audio
        y_inv = self.spectrogram_to_audio(S_db, out_file=out_file)

        # Step 3: Load original for comparison
        y_orig, _ = librosa.load(file_path, sr=self.sr)

        # Step 4: Compute similarity (MSE)
        min_len = min(len(y_orig), len(y_inv))
        mse = np.mean((y_orig[:min_len] - y_inv[:min_len])**2)

        print(f"Round trip verification done! MSE = {mse:.6f}")
        return mse
    
    def detect_onsets(self, file_path="song-track.wav"):
        y, sr = librosa.load(file_path, sr=None)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, units="frames")

        return y,sr,onsets
    
    def spectral_reverb_measure(self, audio_path):
        audio,sr = librosa.load(audio_path)
        # Get amplitude envelope (not spectrogram)
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find the hit - use onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=hop_length)
        
        if len(onset_frames) == 0:
            # Fallback to peak
            peak_frame = np.argmax(rms)
        else:
            peak_frame = onset_frames[0]
        
        # Define windows
        # Direct sound: 0-50ms after hit
        # Reverb tail: 100-400ms after hit
        direct_start = peak_frame
        direct_end = peak_frame + int(0.05 * sr / hop_length)  # 50ms
        
        tail_start = peak_frame + int(0.1 * sr / hop_length)   # 100ms
        tail_end = peak_frame + int(0.4 * sr / hop_length)     # 400ms
        
        # Make sure we don't go out of bounds
        direct_end = min(direct_end, len(rms))
        tail_end = min(tail_end, len(rms))
        
        if tail_start >= len(rms):
            return 0  # Sample too short
        
        # Calculate energies
        direct_energy = np.mean(rms[direct_start:direct_end]**2)
        tail_energy = np.mean(rms[tail_start:tail_end]**2)
        
        # Reverb ratio (higher = more reverb)
        reverb_ratio = tail_energy / (direct_energy + 1e-10)
        
        return reverb_ratio