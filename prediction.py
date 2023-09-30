import librosa
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras


def process_input(audio_file, track_duration):

   

    sample_rate = 22050
    num_mfcc= 13
    n_fft = 2048
    hop_length = 512
    TRACK_DURATION =  track_duration # Replace this with the desired track duration in seconds
    SAMPLES_PER_TRACK = int(sample_rate * TRACK_DURATION)
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

# Assuming you have an audio signal 'signal' and start/finish indices defined


    signal, sample_rate = librosa.load(audio_file, sr=sample_rate)

    for d in range(NUM_SEGMENTS):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            return mfcc
        


genre_dict={0:'blues',1:'classical',2:'country',3:'disco',4:'hiphop',5:'jazz',6:'metal',7:'pop',8:'reggae',9:'rock'}

new_input=process_input('filename',30)      # Replace filename with the name of the audio file you want to predict
X_to_predict=new_input[np.newaxis,...]

model_cnn = keras.models.load_model("music_genre_cnn.h5")

predict=model_cnn.predict(X_to_predict)
predicted_index=np.argmax(predict,axis=1)
print("predicted genre is {}".format(genre_dict[int(predicted_index)]))