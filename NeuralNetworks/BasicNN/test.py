import numpy as np
from onehot_encode import OnehotEncode as OHE
from PIL import Image
import os, time, shutil
from datetime import datetime
import network_gptversion as nw
from audioextract import ExtractAudio as AE
# import network as nw

def train_img():
    input_size = 28 * 28 # 28x28 pixel image dataset
    output_size = 3
    hl_size = 128 # hidden layer size

    nn = nw.NeuralNetwork(input_size,hl_size,output_size)

    # Train for handwritten characters 'O', 'I', and '-'
    characters = []
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                if int(str.split(filename,".")[0]) > 20:
                    characters.append("-")
                elif int(str.split(filename,".")[0]) > 10:
                    characters.append("I")
                else:
                    characters.append("O")
                    
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to match your input size
                images.append(np.asarray(img))
        return np.asarray(images)

    p = r"C:\Users\Kurt\Documents\TrainingData\dataset"
    image_matrix = load_images_from_folder(p)
    print(characters)

    ohe = OHE(characters)
    ohed = ohe.encode()
    print(ohe.labels)

    image_matrix = image_matrix / 255
    image_matrix = image_matrix.reshape(image_matrix.shape[0],-1)

    nn.train(image_matrix,ohed,10000,0.01)

    # Testing
    # h = r"C:\Users\Kurt\Documents\TrainingData\test\-.png"
    # imgw = Image.open(h).convert("L")    
    # imagew_matrix = np.asarray(imgw) / 255.0
    # imagew_matrix = imagew_matrix.reshape(1,-1)
    # prediction = nn.predict(imagew_matrix)
    # print("the predictions is: ", ohe.labels[np.argmax(prediction)])

# Training with audio samples
p = r"C:\Users\Kurt\Music\Muse - Hysteria Instrumental.m4a"
audio_files = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\lmms_samples"
drumsamples = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\WAV"

ae = AE()

audio_spects = [] # Contains a list of spectrograms for each training data audio file/sample
labels = [] # contains the labels of the training data (onehot encoded). The labels list should be in the same order as the training data list.

# loads training data collected by me (from lmms samples lol)
def setup_trainingdata1(filterlist=[],filter_type=None):
    """Blacklist and whitelists are case sensitive (at the moment)\n
    Blacklist is prioritized above whitelist."""

    global audio_spects

    for sound_folder in os.listdir(audio_files):

        # Blacklists or whitelists folders found within training data folder
        if filter_type == "blacklist":
            if sound_folder in filterlist:
                continue
        elif filter_type == "whitelist":
            if not sound_folder in filterlist:
                continue
        
        # Get folder path for each sub folder for individual instrument variants (kick, snare, etc)
        folder_path = (audio_files + "\\" + sound_folder)

        for sounds in os.listdir(folder_path):
            # Get filepath for sound file within sub folder
            filepath = folder_path+"\\"+sounds

            # Create label for one-hot encoding
            labels.append(sound_folder)

            # Extract spectrogram for audio sample
            spec = ae.extract_spectrogram(filepath)
            audio_spects.append(spec.reshape(-1))
    
    audio_spects = np.array(audio_spects)  # shape: (num_samples, input_size)

def setup_trainingdata2(data_folderpath,filterlist={},filter_type=None):
    global audio_spects

    for sound in os.listdir(data_folderpath):
        if sound in filterlist:
            continue

        sound_path = os.path.join(data_folderpath,sound)

        for dirpath, dirs, filenames in os.walk(sound_path):
            for file in filenames:
                filepath = os.path.join(dirpath,file)
                # print(sound,file, filepath)

                labels.append(sound)
                # Extract spectrogram for audio sample
                spec = ae.extract_spectrogram(filepath)
                audio_spects.append(spec.reshape(-1))

    audio_spects = np.array(audio_spects)  # shape: (num_samples, input_size)

print(f"Setting up training data - {datetime.fromtimestamp(time.time())}")
start_time = time.time()
# setup_trainingdata1(["Other"], "blacklist")
setup_trainingdata2(drumsamples,{"Toms","Hihat1"},"blacklist")
print(f"Finished setting up training data. ({time.time() - start_time}s)")

# One-hot encoding
ohe = OHE(labels)
ohed = ohe.encode()
print("Labels:",ohe.labels)

# NN Training
n_mels,max_len = 128,200
input_size = n_mels * max_len
hidden_size = 256
output_size = len(ohe.labels)
nn = nw.NeuralNetwork(input_size,hidden_size,output_size)

nn.load_model("model.npz","config.json")

def train_model(epochs=10000):
    start_time = time.time()
    print(f"\nStarting training at {datetime.fromtimestamp(start_time)}\nEpochs: {epochs}\nLearning Rate: 0.01\nLog Frequency: 500 epochs\n")
    nn.train(audio_spects, ohed, epochs=epochs, learning_rate=0.01,epolog_freq=500)

    duration = time.time() - start_time
    struct = time.gmtime(duration)

    print(f"\nFinished training: {datetime.fromtimestamp(time.time())}\nDuration: {time.strftime("%H:%M:%S", struct)}")

# train_model(10000)

# Testing
t1 = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\testdata\snare_hiphop01.ogg"
t2 = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\testdata\kick03.ogg"
t3 = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\testdata\hihat_closed03.ogg"

def test():
    spec = ae.extract_spectrogram(t1)
    spec_flat = spec.reshape(1, -1)  # batch of 1
    prediction = nn.predict(spec_flat)
    print("Prediction:", ohe.labels[np.argmax(prediction)], prediction)
# test()

sp = r"C:\Users\Kurt\Music\drum_sample.wav"
def detect_onset():
    y,sr,onsets = ae.detect_onsets(sp)
    for onset in onsets:
        feat = ae.extract_spectrogram_patch(y, sr, onset)
        feat_flat = feat.reshape(1, -1)  # batch of 1
        prediction = nn.predict(feat_flat)
        print(f"Onset at {onset * 512 / sr:.2f}s → {ohe.labels[np.argmax(prediction)]} {prediction}")
detect_onset()

def sample_select():
    samples = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\Hihat"
    destination = r"C:\Users\Kurt\Documents\PythonProjects\NeuralNetwork\traindata\WAV\noms"
    nominees = []

    for sample in os.listdir(samples):
        sample_path = os.path.join(samples,sample)
        # print(sample_path)

        rev_rat = ae.spectral_reverb_measure(sample_path)
        if rev_rat <= 0.02:
            print(sample, rev_rat)
            nominees.append(sample)

            shutil.copy2(sample_path,destination)

    print("finished",len(nominees))