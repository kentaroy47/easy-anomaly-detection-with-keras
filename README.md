# What is this repo?
- **It's Simple**

Learn from the basics of anomaly detection. How it works and analyze.

- **Train real Deep Learning Models**

Learn to construct time-series models with fully connected or LSTMs or GRU cells.

- **It's Easy**

It's written with Keras, easy to understand.




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
## data and model setups
![anomal](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/figs/norm.JPG)

we split the long train data and test data into sequences which has 400 data samples each.

our keras model is trained with tons of *normal* data, and trained to predict how the *next* sequence looks like.

![predict](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/figs/normal_waveform_predict.png)

by training an autoencoder, DNNs do well learning and generating data. (the training data is toy data and is very easy!)

BTW. We trained this model by:

```
python anomaly_withFC.py --epoch 100
```

## anomaly detection
![results](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/figs/FC_waveforms.png)

Let's see the how the model behaves with normal and anomaly data.

The data 0-2500 is normal similar to the trained state, and 2500~ is anomaly (with large amplitudes.).

The **model results and the real data has no contradictions (data 0~2500), if the state is normal.**

The model results and the real data **have large differences (data 2500-), the input data is clearly different from the normal trained data.** 

When the model cannot predict the next state, and we define that it is likely to be an anomaly state.

we plot the anomaly score as bellow (which is the square difference between the predicted and real data)

by looking at this, **we can find when and how long anomaly events have occured.**

![results](https://github.com/kentaroy47/anomaly-detection-with-keras/blob/master/figs/FC_anomaly_score.png)

# running the training scripts
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

