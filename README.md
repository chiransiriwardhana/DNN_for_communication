# OFDM Signal Detection using Deep Neural Network (DNN)
This repository implements a deep neural network (DNN) to jointly estimate the channel and detect transmitted bits in an OFDM system, replicating and improving the results in the paper:

H. Ye, G. Y. Li, and B.-H. Juang,
“Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems,”
IEEE Wireless Communications Letters, vol. 7, no. 1, pp. 114–117, Feb. 2018.

### Features
	•	BPSK-modulated OFDM system with 64 subcarriers
	•	Nonlinear distortion modeled with tanh
	•	Pilot-based MMSE channel estimation
	•	Deep Neural Network (DNN) for joint detection
	•	BER comparison with MMSE and LS estimators
	•	BER vs SNR curve generation
	•	Constellation plotting of received symbols


### Model Architecture
	The DNN model has:
	•	Input: Real & Imag parts of subcarriers and pilot symbols
	•	Hidden Layers: [512 → BatchNorm → 256 → BatchNorm → 128]
	•	Output: 64 sigmoid neurons for bit-wise classification

Loss: Binary Crossentropy
Optimizer: Adam
Epochs: 10
Batch Size: 128

### Requirements

#### Install dependencies:
pip install numpy tensorflow scikit-learn matplotlib
#### Run the Code
python DNN_for_communication.py
### Key Functions
	•	bpsk_mod(bits): Converts bits to BPSK symbols.
	•	add_awgn_noise(signal, SNR_dB): Adds Gaussian noise to signal.
	•	mmse_channel_est(pilots_rx, pilots_tx): Simple MMSE estimator using pilots.
	•	apply_nonlinear_distortion(signal): Simulates RF nonlinearities.
	•	model.fit(...): Trains DNN on pilot-enhanced features.

 ### Performance Evaluation
	•	Uses train_test_split() to partition data.
	•	Evaluation metrics:
	•	Bitwise Accuracy (np.mean(y_pred == y_true))
	•	Bit Error Rate (np.mean(y_pred != y_true))

 ### Visualizations
	•	Constellation Plot of received OFDM symbols
	•	BER vs. SNR Curve for DNN, MMSE, and LS detectors
