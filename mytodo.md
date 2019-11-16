# Sound2

## 1. Project
Next step in the V2 sound classifier: train the model on a new sound category, find out how many samples are needed

## 2. Steps

### 2.1 Improve current accuracy [4 h]
 * Frequency is not considered in current version: different recording frequencies now lead to stretched pectograms. See what happens when the frequency is taken into account. Try loading the sounds with Librosa [like in this tutorial](https://towardsdatascience.com/urban-sound-classification-part-2-sample-rate-conversion-librosa-ba7bc88f209a)
 * Try different spectrogram, try the code in [this tutorial](https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7)

### 2.2 Accuracy vs # of samples [4 h]
 * train from scratch with different fractions of the data and see the accuracy vs # of samples

### 2.3 Add category [8 h]
 * train model further with new category, with different fractions of available samples to see the effect on accuracy

### 2.4 Realtime classification tool [6 h]
 * Build tool
 * make standalone / easily deployable on Raspberry Pi / Mac / Linux / Windows. [Docker?](https://sigmoidal.io/how-to-reuse-keras-deep-neural-network-using-docker/)

### 2.5 Sharing [3 h]
 * Clean up repo, publish on github
 * Write doumentation (or at least an extensive readme.md)

## 3. Hours
Total allocated hours: 20

Hours spent prior to these steps:
 * data inspection, requirements determination for samples: 1 h
 * Collect sounds (my part 1/3): 1.5 h
 * Writing this plan, research on possibilities and existing tutorials: 1.5 h

Total 4 h, 16 h left for steps
 > Problem: estimated total = 29 h, 13 hours short.
