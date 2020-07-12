# Speech_Command_Recognition

# Problem Description 
The goal of this project is to convert single word speech commands into text. There are a lot of applications for speech recognition today with almost every single device coming in with an inbuilt speech recognition software for instance Alexa, Google home and Siri. I have taken a few speech commands and classified them into text. This serves the purpose of identifying when single word commands are spoken and converting those to text. I have used classification algorithms to classify the words into text classes. 

# Data Description  
The data used is the official Speech Commands Dataset provided by TensorFlow. This dataset includes 95,000 one second long utterances of 35 short words, by thousands of people. The audio files were collected using crowdsourcing where anyone can contribute to this data.  The total size of the data is 2.3 GB compressed and it is divided into folders for each word. The words are simple speech commands people speak (forward, backward, up, down, yes, no, zero, one, two, etc.). There is also a folder in there which includes some audio files containing common background noise (white noise, washing dishes, etc.). 
For this project, I have trimmed down the data to consist of 12 categories. Out of the 35 commands, the 
10 commands I have chosen are “forward”, “backward”, “yes”, “no”, “off”, “on”, “up”, “down”, “happy” and “stop”. I have also created a category called “unknown” in which I have fed data for “zero” and “nine”. This category allows to club various categories into one. There is also a category created called “silence” which includes the background noise data. This helps to identify cases where no commands are spoken and there is just some background noise (for instance white noise). In total, I have considered 41,000 audio files.  

Dataset: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz 

# Data Preparation 
I took multiple steps to prepare the data: - 
1. Resampling – 
The original data obtained was sampled at 16,000 Hz. I re-sampled the data to 8,000 Hz since it was not losing a lot of accuracy with the down sampling. It also helped reduce the size of the data by half which was really helpful when loading the data and processing it. 
2. Data Slicing – 
The original data has a “_background_noise_” folder which contains 1-minute long audios of general background noises. Since the model is working with 1 second audio, I sliced the audio files in the “_background_noise_” folder to 1 second audio clips as the input length should be one second. 
3. Adding background noise –
In order to mimic more real-world scenarios for speech, we decided to add background noise to a few audio files for every word. We randomly chose 500 words from each folder and added a random background noise to them (obtained from the background noise files I had in the original data).  
4. Speed Tuning – 
Speed tuning involves changing the speed of the spoken word. I chose 200 random files for each word and applied speed tuning to them. This would either make the word be spoken a little slower or a little faster. Again, this helps to introduce some variation in the data to mimic the real world. 

# Code folder structure: 
The code is modularized, and I have split the code up into multiple logical files. Here is a reference to the files in the folder: - 
-	1_Audio_preprocessing.ipynb – 
This contains code for all the preprocessing that was performed on the data. It contains code for resampling the data from 16Khz to 8Khz. It also has logic to slice the background noise files into 1-second audio clips. 
-	2_Audio_Data_Visualization.ipynb – 
This has all the visualization logic that I wrote. It contains the visualization of the audio and also the Spectrogram. I also show how we can visualize MFCC. This file also contains the visualization showing the majority class. “Unknown” is the majority class. I also have visualization depicting the different durations of the recordings. 
- 3_Audio_Augmentation.ipynb – 
The first thing I have shown here is how to read and play an audio. I am augmenting data here by adding a random background noise (chosen from the background noise data downloaded with the original dataset) to 500 audio files in every data folder. As a next step, I performed some speed tuning as well. This either slows down or speeds up the audio. This is done so that I can give varied data to the network and also emulate the real-world scenario. 
-	4_Audio_Model1_Conv1D.ipynb – 
This is the first model which works with the raw data and uses CNN 1D. 
-	5_Audio_Model2_Conv2D.ipynb – 
In this model, I have used Convolution 2D and MFCC feature extraction.
-	6_Audio_Model3_Attention_RNN.ipynb – 
This is the third model which is based on RNN and uses Attention Mechanism. 
-	7_Create_your_own_recording.ipynb – 
This gives us the ability to load a saved model (either of the 3 models) and record our own 1 second audio clip. There is code after this which will do the prediction and classify the audio into its class. 
- 8_Model_Architecture_Visualization.ipynb – 
This contains the visualization for all the 3 models I have built. It shows all the layers in the models. 



