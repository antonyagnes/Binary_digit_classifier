# Binary_digit_classifierProblem 2

Defining the problem
	Audio analysis is one of the growing areas since the development of Deep Learning.
The problem is to build a binary classifier to identify two different digits.

Objectives

	Use the .wav files to identify the digit. The digits are for three different speakers. We say 1 for digit 2 and 0 for digit 1.

Install 

Librosa
Glob
Radom
Numpy
Keras
Pandas

Run

	In the terminal or window navigate to the top level of the project directory and run the following command
python digit_classifier.py

Data

A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz. The recordings are trimmed so that they have near minimal silence at the beginnings and ends.
FSDD is an open dataset, which means it will grow over time as data is contributed. Thus in order to enable reproducibility and accurate citation in scientific journals the dataset is versioned using git tags.
Current status
•	3 speakers
•	1,500 recordings (50 of each digit per speaker)
•	English pronunciations

Target variable

1 for digit 2
0 for digit 1
