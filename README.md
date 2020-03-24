# eeg-time-series-classification

The objective here is the classify the action of a human subject, given 2115 trials of EEG time series data from 22 electrodes, from 9 different people. I'm attempting several as simple as possible architectures and evaluating which one performs best.

* Combination_Model.ipynb has the 3 layer CNN. This model had the best performance

* LSTM_FCN_Combination.ipynb has LSTM and CNN trained together and combined (summed). This model had the second best performance.

* RNN_CNN_Combination.ipynb has an RNN followed by the same CNN architecture in Combination Model

* Parallel_LSTM_CNN.ipynb has 5 LSTMs trained on 200 time points of data each. This architecture performed terribly.

* LSTM_Combination_Model.ipynb has an LSTM followed by fully connected layers. This architecture was also quite bad.

Models were trained in Colab, with GPU support.
