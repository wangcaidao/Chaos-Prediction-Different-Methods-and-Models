# Chaos-Prediction-Different-Methods-and-Models
Using Different Methods and Models to predict chaotic time series

This project provides various time series forecasting methods (P2P, NAR, Seq2Seq) and different deep learning models (MLP, LSTM, FNO) for short-term prediction and long-term modeling of time series.

## Requirements
- Python 3.7
- Pytorch 1.8.0 or later
- tqdm
- scipy
- numpy

My environment has been exported to `environment.txt`.

## Usage:
1. **Prepare Time Series Data**
   Use the Rossler/Lorenz/MG/Ikeda_generator.m scripts under `/data` to generate time series, or use your own prepared time series. Ensure the variable name is `u` and the shape is `(time_steps x feature_dim)`. Save the data in the `Time Series` folder.

2. **Create Training and Testing Datasets for Different Methods**
   Use the `gen_dataset_XXX.m` scripts under `/data` to generate datasets for different methods. After running, three files will be created: the training dataset, short-term testing dataset, and long-term testing dataset (except for `gen_dataset_onestep`). The generated datasets are saved in `.mat` format in the `Datasets` folder.

3. **Train Deep Learning Models**
   Use the `train_XXX.py` scripts in the root directory to train models using different methods. You can modify the model used by changing the `model` variable in the code. Ensure that the `in_dim` and `out_dim` defined in the model match the input and output dimensions. To train the model, set `i=1` in the main function. We use `/configs/XXX/train.yaml` to set various hyperparameters.

4. **Evaluate Training Results**
   After training, to test the trained model, set `i=2` in the main function. We use `/configs/XXX/test.yaml` to set various hyperparameters. Testing includes short-term prediction and long-term modeling, which can be switched by changing the files used in the `.yaml` configuration. After running, the model's output will be saved as `.mat` files in the `/pred` folder. Then, use the following scripts in the root directory to visualize the results:
   - `plot_NonDelay`: Evaluate short-term prediction results for non-delayed sequences.
   - `plot_NonDelay_phase`: Evaluate long-term prediction results for non-delayed sequences (plot phase diagrams).
   - `plot_TimeDelay`: Evaluate short-term prediction results for delayed sequences.
   - `plot_TimeDelay_phase`: Evaluate long-term prediction results for delayed sequences (plot phase diagrams).

**Additional: About Onestep**
We also provide code for one-step prediction training and evaluation (only applicable to S2S-1P and S2S-TD). After preparing the time series, use `gen_dataset_onestep.m` to create the dataset, `train_onestep.py` for training and testing, and `plot_OneStep` to view the test results.
