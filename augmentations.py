import os
import librosa
import random
from audiomentations import Compose, PitchShift, AddGaussianNoise, HighPassFilter, LowPassFilter
from audiomentations import Padding, TimeStretch, Gain
import soundfile as sf
from glob import glob

# Define augmentation transformations
def get_random_augmentation():
    return Compose([
        PitchShift(
            min_semitones=random.uniform(-2, 0), 
            max_semitones=random.uniform(0, 2), 
            p=random.uniform(0, 1)
        ),
        HighPassFilter(
            min_cutoff_freq=random.uniform(300, 700), 
            max_cutoff_freq=random.uniform(700, 1500), 
            p=random.uniform(0, 1)
        ),
        LowPassFilter(
            min_cutoff_freq=random.uniform(2000, 2200), 
            max_cutoff_freq=random.uniform(2200, 3000), 
            p=random.uniform(0, 1)
        ),
        Padding(
            mode="silence", 
            max_fraction=random.uniform(0.1, 0.5), 
            pad_section=random.choice(["start", "end"]),
            p=random.uniform(0, 1)
        ),
        TimeStretch(
            min_rate=random.uniform(0.8, 1.0), 
            max_rate=random.uniform(1.0, 1.2), 
            leave_length_unchanged=random.choice([True, False]), 
            p=random.uniform(0, 1)
        ),
        Gain(
            min_gain_db=random.uniform(1, 5), 
            max_gain_db=random.uniform(5, 10), 
            p=random.uniform(0, 1)
        ),
        AddGaussianNoise(
            min_amplitude=random.uniform(0.0005, 0.001), 
            max_amplitude=random.uniform(0.001, 0.005), 
            p=random.uniform(0, 1)
        ),
    ])

# List of audio class types
audio_types = ["belly_pain", "burping", "cold-hot", "discomfort", "dontKnow" , "hungry", "lonely", "scared", "tired"]

# Directory paths
input_dataset = "E:\Senior_Project\Sound_Detection\seniorDataset"
output_dataset = "aug-seniorDataset"

# Create the main directory for augmented files
os.makedirs(output_dataset, exist_ok=True)

# Step 1: Determine the largest class size and cap it at 200 files
max_files = 210

# Step 2: Augment each class to match the maximum file limit of 200
for audio_type in audio_types:
    # Get all the audio files for the current class
    input_class_path = f"{input_dataset}/{audio_type}"
    
    # Verify if the directory exists
    if not os.path.exists(input_class_path):
        print(f"Directory {input_class_path} does not exist! Skipping this class.")
        continue

    audio_files = glob(f"{input_class_path}/*.wav")
    num_files = len(audio_files)

    # Print the number of files found for debugging
    print(f"Found {num_files} files in {input_class_path}")

    # If the number of files is already 200 or more, skip augmentation
    if num_files >= max_files:
        print(f"Skipping augmentation for {audio_type} as it already has {num_files} files.")
        continue

    # Calculate how many files need to be generated to reach 200
    num_to_generate = max_files - num_files

    # Create a subfolder for each audio class inside the output folder
    output_class_path = f"{output_dataset}/{audio_type}"
    os.makedirs(output_class_path, exist_ok=True)

    # Copy existing files to the output folder
    for file in audio_files:
        try:
            # Load the audio file
            signal, sr = librosa.load(file, sr=None)  # Preserve original sample rate
            file_name = os.path.basename(file)
            
            # Save the original file in the output folder
            sf.write(f"{output_class_path}/{file_name}", signal, sr, subtype='PCM_16')
        except Exception as e:
            print(f"Error copying file {file}: {e}")

    # Generate augmented files
    for i in range(num_to_generate):
        try:
            # Randomly select a file to augment
            random_file = random.choice(audio_files)
            signal, sr = librosa.load(random_file, sr=None)
            
            # Apply augmentation
            augmentation = get_random_augmentation()
            augmented_signal = augmentation(signal, sr)
            
            # Save the augmented file
            random_file_name = os.path.basename(random_file).replace(".wav", "")
            augmented_file_name = f"aug-{random_file_name}-{i}.wav"
            sf.write(f"{output_class_path}/{augmented_file_name}", augmented_signal, sr, subtype='PCM_16')
        except Exception as e:
            print(f"Error augmenting file {random_file}: {e}")
