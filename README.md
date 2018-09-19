# anomaly-detection-with-keras
Lets do anomaly detection with keras!

# to get started
clone the repo.
```
git clone https://github.com/kentaroy47/anomaly-detection-with-keras.git
```

## for fully connected cells..
```
python anomaly_withFC.py
```

## for LSTM cells..
```
python anomaly_withLSTM.py
```

## for GRU cells..
```
python anomaly_withGRU.py
```

### files
https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/norm.JPG?raw=true
normal.csv includes artificial waveform that is used for training.
The NN will think that this is the normal state.
WIth an Autoencoder, the NN is trained to reproduce this state.

abn.csv includes abnormal data for validation.
The basic thinking of anomal detection is that *autoencoders cannot reproduce untrained data*.
Therefore, if the autoencoders cannot reproduce data correctly, it is likely that the data is anomal (or unseen data).

