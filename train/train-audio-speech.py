import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kult.utils.feature_extraction import get_features_file
from kult.models.audio_model import CNN1D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
training_runs_folder = os.path.join(script_dir, "training_runs")
training_folder = os.path.join(training_runs_folder, timestamp)
os.makedirs(training_folder, exist_ok=True)

# Processing CREMA Dataset

CREMAD_PATH = os.path.join(script_dir, "datasets/CREMA-D")
CREMAD_PATH_WAV = os.path.join(CREMAD_PATH, "AudioWAV")

# Creating dictionary from CREMA github documentation on actors' gender.
Crema_genders = pd.read_csv(os.path.join(CREMAD_PATH,'VideoDemographics.csv'))
select_cond = ['ActorID', 'Sex']

# Looks like this: {1001: 'female'}
extraction = Crema_genders[select_cond]
extraction_dict = dict(extraction.values)

emotion = []
intensity = []
gender = []
location = []

wavs = [x for x in os.listdir(CREMAD_PATH_WAV) if not x.startswith('.')]

for wav in wavs:
    partition = wav.split('_')
    emotion_ = partition[2]
    intensity_ = partition[3].split('.')[0]
    actor_ID = int(partition[0])
    
    # looks up the dictionary made in prev cell
    if actor_ID in extraction_dict: gender.append(extraction_dict.get(actor_ID))
    
    intensity.append(intensity_)
    emotion.append(emotion_) 
    location.append(CREMAD_PATH_WAV + '/' + wav)

crema_df = pd.DataFrame({'gender': gender, 'emotion': emotion, 'location': location}, 
                        columns=['gender','emotion', 'location'])

crema_df = crema_df.replace({'emotion': {'ANG':'angry','DIS': 'disgust',
                                        'FEA': 'fear','HAP': 'happy',
                                        'NEU': 'neutral','SAD': 'sad'}})

crema_df = crema_df.replace({'intensity': {'LO': 'low','MD': 'medium',
                                            'HI': 'high','XX': 'unspecified'}})

combined_df = pd.concat([crema_df, ], axis=0)

print(combined_df.emotion.value_counts())
print('Dataset shape =', combined_df.shape)

# Note: The audio file in cremad/AudioWAV/1076_MTI_SAD_XX.wav is labeled as sad but it is actually an empty audio file
to_delete = combined_df.index[combined_df['location'] == os.path.join(CREMAD_PATH_WAV, '1076_MTI_SAD_XX.wav')].tolist()
combined_df = combined_df.drop(to_delete)

audio_features_list = []
genders_list = []
emotions_list = []
locations_list = []

file_path_parquet = os.path.join(training_runs_folder, "audio_features.parquet")
if os.path.exists(file_path_parquet):
    audio_features_df = pd.read_parquet(file_path_parquet)
else:
    print('Audio Data - Features Extraction Starts')

    for gender, emotion, file_path, index in zip(combined_df.gender, combined_df.emotion, combined_df.location, tqdm(range(combined_df.location.shape[0]))):
        audio_features = get_features_file(file_path)    
        
        if index % 1 == 0:
            print(index, ' audio files have been feature extracted(', round(index / 12162 * 100), '%)\r', sep = '', end = '')
        
        for i in audio_features:
            audio_features_list.append(i)
            genders_list.append(gender)
            emotions_list.append(emotion)
            locations_list.append(file_path)

    print('\nAudio Data - Features Extraction Ends')

    audio_features_tuple = tuple(audio_features_list)

    audio_features_df = pd.DataFrame(audio_features_tuple)

    audio_features_df['gender'] = genders_list
    audio_features_df['emotion'] = emotions_list
    audio_features_df['location'] = locations_list

    audio_features_df.columns = audio_features_df.columns.astype(str)
    audio_features_df.to_parquet(file_path_parquet, engine = 'fastparquet')

audio_features_df = audio_features_df.fillna(0)
audio_features_df = audio_features_df.drop_duplicates()

#

X = audio_features_df.drop(labels = ['emotion', 'gender', 'location'], axis = 1)
y = audio_features_df['emotion']

label = LabelEncoder()
y = label.fit_transform(y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size = 0.2, random_state = 42)

standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train.values)
X_valid = standard_scaler.transform(X_valid.values)

scaler_filename = os.path.join(training_folder, 'standard_scaler.save')
joblib.dump(standard_scaler, scaler_filename)

# Modify data to fit into a 1D-CNN.
X_train = np.expand_dims(X_train, axis = 2)
X_valid = np.expand_dims(X_valid, axis = 2)

#

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

num_classes = len(label.classes_)
model = CNN1D(input_shape=X_train.shape[1], num_classes=num_classes).to(device)

# Callback-like функции: EarlyStopping и ReduceLROnPlateau
class EarlyStopping:
    def __init__(self, patience=10, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_model = None

    def __call__(self, current_score, model):
        if self.best_score is None or \
            (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score
            self.best_model = model.state_dict()
            self.counter = 0
            print(f'Best model updated with score: {self.best_score:.4f}')
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True  # Stop training
        return False

# Задаем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=0.000001)
scheduler_start = 90
early_stopping = EarlyStopping(patience=20, mode='min')

# Тренировка
num_epochs = 1000
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

print()
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.squeeze(-1).unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs.squeeze(-1).unsqueeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(valid_loader)
    val_accuracy = 100 * correct / total

    epoch_time = time.time() - epoch_start_time
    total_time_elapsed = time.time() - start_time
    estimated_total_time = (total_time_elapsed / (epoch + 1)) * num_epochs
    estimated_remaining_time = estimated_total_time - total_time_elapsed

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, "
          f"Time/Epoch: {epoch_time:.2f}s, Elapsed: {total_time_elapsed:.2f}s, Remaining: {estimated_remaining_time:.2f}s")

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['accuracy'].append(train_accuracy)
    history['val_accuracy'].append(val_accuracy)

    # Использование ReduceLROnPlateau для динамического уменьшения learning rate
    if (val_accuracy > scheduler_start):
        scheduler.step(val_loss, epoch=epoch+1)

    # Check for early stopping
    if early_stopping(val_loss, model):
        model.load_state_dict(early_stopping.best_model)
        break

# Сохранение модели
torch.save(model.state_dict(), os.path.join(training_folder, 'speech_model.pth'))

# Сохранение истории
with open(os.path.join(training_folder, 'trainHistoryByEpoch.txt'), 'w') as history_file:
    for epoch in range(len(history['loss'])):
        # Форматируем строку для текущей эпохи
        epoch_data = (
            f"Epoch: {epoch + 1}, "
            f"Loss: {history['loss'][epoch]:.4f}, "
            f"Val Loss: {history['val_loss'][epoch]:.4f}, "
            f"Accuracy: {history['accuracy'][epoch]:.2f}%, "
            f"Val Accuracy: {history['val_accuracy'][epoch]:.2f}%\n"
        )
        # Записываем строку в файл
        history_file.write(epoch_data)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(history['loss'], label='loss')
ax1.plot(history['val_loss'], label='val_loss')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Epoch')
ax1.grid(True)

ax2.plot(history['accuracy'], label='accuracy')
ax2.plot(history['val_accuracy'], label='val_accuracy')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Epoch')
ax2.grid(True)

fig.set_figheight(10)
fig.set_figwidth(35)
plt.tight_layout()
plt.savefig(os.path.join(training_folder, 'training_history.png'))