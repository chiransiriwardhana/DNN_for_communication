# DNN_for_communication
This repository implements a deep neural network (DNN) to jointly estimate the channel and detect transmitted bits in an OFDM system, replicating and improving the results in the paper:

H. Ye, G. Y. Li, and B.-H. Juang,
“Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems,”
IEEE Wireless Communications Letters, vol. 7, no. 1, pp. 114–117, Feb. 2018.

### Features
	•	BPSK modulation over 64-subcarrier OFDM.
	•	Multipath Rayleigh fading channels.
	•	AWGN noise and nonlinear distortion simulation.
	•	Pilot-assisted MMSE and LS estimators.
	•	DNN model trained offline using pilots + received data.
	•	BER and bit accuracy comparison.

### Model Architecture
	•	Input: Real and imaginary parts of received symbols + pilot symbols.
	•	Architecture:
	•	Dense(512) → BatchNorm → Dropout(0.3)
	•	Dense(256) → BatchNorm → Dropout(0.3)
	•	Dense(128)
	•	Dense(64) output (1 per bit)
	•	Output: Bitwise logits (interpreted using sigmoid during inference).
