import librosa
import numpy as np
import soundfile as sf

class ExtractAudio:
    def __init__(self, n_fft=2048, hop_length=512, max_len=None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = max_len
        self.sr = None

    def extract_stft(self, file_path):
        """Return complex STFT matrix (keeps phase for exact reconstruction)."""
        y, self.sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)

        # Optionally pad/truncate time frames
        if self.max_len:
            if D.shape[1] < self.max_len:
                pad_width = self.max_len - D.shape[1]
                D = np.pad(D, ((0,0),(0,pad_width)), mode="constant")
            else:
                D = D[:, :self.max_len]

        return D

    def stft_to_audio(self, D, out_file=None):
        """Reconstruct audio exactly from complex STFT."""
        y_inv = librosa.istft(D, hop_length=self.hop_length)

        if out_file:
            sf.write(out_file, y_inv, self.sr)
            print(f"Reconstructed audio saved to {out_file}")

        return y_inv

    def verify_round_trip(self, file_path, out_file="roundtrip_exact.wav"):
        """Verify that STFT round trip is exact."""
        # Step 1: STFT
        D = self.extract_stft(file_path)

        # Step 2: Invert
        y_inv = self.stft_to_audio(D, out_file=out_file)

        # Step 3: Compare with original
        y_orig, _ = librosa.load(file_path, sr=self.sr)

        # Ensure equal length
        min_len = min(len(y_orig), len(y_inv))
        mse = np.mean((y_orig[:min_len] - y_inv[:min_len])**2)

        print(f"Round trip verification done! MSE = {mse:.10f}")
        return mse
    
    def extract_features(self, file_path):
        y, self.sr = librosa.load(file_path, sr=None)
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)  # magnitude

        # Convert to log scale
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # Pad or truncate to fixed length
        if S_db.shape[1] < self.max_len:
            pad_width = self.max_len - S_db.shape[1]
            S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            S_db = S_db[:, :self.max_len]

        # Normalize 0–1
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

        return S_norm  # shape (freq_bins, time_frames)
    
    def detect_onsets(self, file_path="song-track.wav"):
        y, sr = librosa.load(file_path, sr=None)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, units="frames")

        return onsets
    

ad = ExtractAudio(max_len=1000)

p = r"C:\Users\Kurt\Music\Muse - Hysteria Instrumental.m4a"
q = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\WAV\Hihat\hihat aasimonster (6).wav"
w = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\WAV\Hihat\hihat aasimonster (18).wav"


# Run exact round trip
# mse = ad.extract_features(p)
# print(mse.shape,mse)

# if mse < 1e-10:
#     print("✅ Perfect reconstruction achieved!")
