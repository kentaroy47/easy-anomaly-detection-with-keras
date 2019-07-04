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

# how anomaly detection works.
![anomal](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/norm.JPG)
we split the long train data and test data into sequences which has 400 data samples each.

our keras model is trained with tons of *normal* data, and trained to predict how the *next* sequence looks like.

![predict](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/normal_waveform_predict.png)

by training an autoencoder, DNNs do well learning and generating data. (the training data is toy data and is very easy!)

BTW. We trained this model by
```
python anomaly_withFC.py --epoch 100
```

![results](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/FC_waveforms.png)

Let's see the model prediction results and the real data.

If the **model results and the real data has no contradictions (data 0~2000), the state is normal!**

if the model results and the real data **have large differences (data 2500-), the input data is clearly different from the normal trained data.** 

The model cannot predict the next state, and we define that it is likely to be an anomaly state.

we plot the anomaly score as bellow (which is the square difference between the predicted and real data)

by looking at this, **we can find when and how long anomaly events have occured.**

![results](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/FC_anomaly_score.png)

## for fully connected cells..
全結合ネットワークでやる場合。。
```
python anomaly_withFC.py
```

### since the model predicts sequential states, it will do better with *reccurent networks* like LSTM and GRUs.

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

# File setup
normal.csv includes artificial waveform that is used for training.

**When using your own data, make sure that this is the normal state.**

WIth an Autoencoder, the NN is trained to reproduce this state.

abn.csv includes **abnormal data for validation.**

The basic thinking of anomal detection is that *autoencoders cannot reproduce untrained data*.

Therefore, if the autoencoders cannot reproduce data correctly, it is likely that the data is anomal (or unseen data).

