import io
import os
import wave
import nltk
import moviepy.editor as mp
import deepspeech

# Set up NLTK for text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Set up the video file
video_file = r'C:\Users\coool\Desktop\NLP project\Coldplay - Hymn For The Weekend (Official Video).mkv'

# Extract audio from video
video = mp.VideoFileClip(video_file)
audio = video.audio
audio.write_audiofile('audio.wav')

# Set up the audio file
audio_file = 'audio.wav'
with wave.open(audio_file, 'rb') as audio:
    audio_content = audio.readframes(audio.getnframes())

# Set up the speech recognition model
model_file_path = 'deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
ds = deepspeech.Model(model_file_path)
ds.enableExternalScorer(scorer_file_path)

# Transcribe the audio file
transcription = ds.stt(audio_content)

# Preprocess the text
tokens = word_tokenize(transcription)
tokens = [token.lower() for token in tokens if token.isalpha()]
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if not token in stop_words]
text = ' '.join(tokens)

# Generate subtitles
subtitles = ''
sentences = nltk.sent_tokenize(text)
for i, sentence in enumerate(sentences):
    start_time = i * 3  # Set the start time of each subtitle to 3 seconds
    end_time = (i + 1) * 3  # Set the end time of each subtitle to 3 seconds
    subtitle = f'{i+1}\n{sentence}\n{start_time},000 --> {end_time},000\n\n'
    subtitles += subtitle

# Output subtitles
with io.open('subtitles.srt', 'w', encoding='utf-8') as f:
    f.write(subtitles)
