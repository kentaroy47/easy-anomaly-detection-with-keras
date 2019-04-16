![anomal](https://cdn-images-1.medium.com/max/1600/1*ZlN46eNWkRtkAS4qOjrJYA.png)
# anomaly-detection-with-keras
Lets do anomaly detection with keras!
Kerasを使って異常検知をしてみましょう！

## Qiitaの解説記事
TBD

# to get started
clone the repo.
レポをクローンする。

```
git clone https://github.com/kentaroy47/anomaly-detection-with-keras.git
```

## for fully connected cells..
全結合ネットワークでやる場合。。
```
python anomaly_withFC.py
```

## for LSTM cells..
LSTMネットワークでやる場合。。
```
python anomaly_withLSTM.py
```

## for GRU cells..
GRUネットワークでやる場合。。
```
python anomaly_withGRU.py
```

### files
![anomal](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/norm.JPG)

normal.csv includes artificial waveform that is used for training.
The NN will think that this is the normal state.
WIth an Autoencoder, the NN is trained to reproduce this state.

abn.csv includes abnormal data for validation.
The basic thinking of anomal detection is that *autoencoders cannot reproduce untrained data*.
Therefore, if the autoencoders cannot reproduce data correctly, it is likely that the data is anomal (or unseen data).

