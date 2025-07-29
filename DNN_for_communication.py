import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulation Parameters
N = 64
num_symbols = 20000
pilot_idx = np.arange(0, N, 4)

def bpsk_mod(bits):
    return 2 * bits - 1

def add_awgn_noise(signal, SNR_dB):
    SNR = 10**(SNR_dB / 10)
    noise_power = np.mean(np.abs(signal)**2) / SNR
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def mmse_channel_est(pilots_rx, pilots_tx):
    H_est = pilots_rx / (pilots_tx + 1e-8)
    return np.interp(np.arange(N), pilot_idx, H_est)

def apply_nonlinear_distortion(signal):
    return np.tanh(signal)

# Dataset Generation
X, Y, Y_mmse, Y_ls = [], [], [], []
SNR_dBs = np.random.uniform(5, 25, num_symbols)

for i in range(num_symbols):
    tx_bits = np.random.randint(0, 2, size=N)
    tx_symbols = bpsk_mod(tx_bits)
    tx_ofdm = apply_nonlinear_distortion(np.fft.ifft(tx_symbols))

    h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    rx_ofdm = add_awgn_noise(tx_ofdm * h, SNR_dBs[i])
    rx_symbols = np.fft.fft(rx_ofdm)

    pilots_tx = tx_symbols[pilot_idx]
    pilots_rx = rx_symbols[pilot_idx]

    H_mmse = mmse_channel_est(pilots_rx, pilots_tx)
    H_ls = H_mmse.copy()

    rx_detect_mmse = (np.real(rx_symbols / (H_mmse + 1e-8)) > 0).astype(np.float32)
    rx_detect_ls = (np.real(rx_symbols / (H_ls + 1e-8)) > 0).astype(np.float32)

    inp = np.stack([np.real(rx_symbols), np.imag(rx_symbols)], axis=1).flatten()
    inp = np.concatenate([inp, np.real(pilots_tx), np.imag(pilots_tx)])

    X.append(inp)
    Y.append(tx_bits.astype(np.float32))
    Y_mmse.append(rx_detect_mmse)
    Y_ls.append(rx_detect_ls)

X = np.array(X)
Y = np.array(Y)
Y_mmse = np.array(Y_mmse)
Y_ls = np.array(Y_ls)

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test, Y_mmse_train, Y_mmse_test, Y_ls_train, Y_ls_test = train_test_split(
    X, Y, Y_mmse, Y_ls, test_size=0.2, random_state=42
)

# DNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(N, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Training
model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)

# Bit Accuracy
y_pred_dnn = model.predict(X_test) > 0.5
bit_accuracy = np.mean(y_pred_dnn == Y_test)
print(f"\n✅ DNN Bit Accuracy: {bit_accuracy * 100:.2f}%")

# BER Computation
ber_dnn = np.mean(y_pred_dnn != Y_test)
ber_mmse = np.mean(Y_mmse_test != Y_test)
ber_ls = np.mean(Y_ls_test != Y_test)

print("\nBER Comparison:")
print(f"DNN BER   : {ber_dnn:.4f}")
print(f"MMSE BER  : {ber_mmse:.4f}")
print(f"LS BER    : {ber_ls:.4f}")

# Constellation Plot
plt.figure(figsize=(6, 6))
rx_symbols_test = X_test[:, :2*N].reshape(-1, 2)
plt.scatter(rx_symbols_test[:, 0], rx_symbols_test[:, 1], s=1, alpha=0.5)
plt.title("Received Symbols Constellation")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.grid(True)
plt.show()

# BER vs SNR
SNR_range = np.arange(0, 30, 2)
ber_dnn_curve, ber_mmse_curve, ber_ls_curve = [], [], []

for snr in SNR_range:
    X_snr, Y_snr, Y_mmse_snr, Y_ls_snr = [], [], [], []
    for _ in range(500):
        tx_bits = np.random.randint(0, 2, size=N)
        tx_symbols = bpsk_mod(tx_bits)
        tx_ofdm = apply_nonlinear_distortion(np.fft.ifft(tx_symbols))
        h = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        rx_ofdm = add_awgn_noise(tx_ofdm * h, snr)
        rx_symbols = np.fft.fft(rx_ofdm)

        pilots_tx = tx_symbols[pilot_idx]
        pilots_rx = rx_symbols[pilot_idx]
        H_mmse = mmse_channel_est(pilots_rx, pilots_tx)
        H_ls = H_mmse.copy()

        rx_mmse = (np.real(rx_symbols / (H_mmse + 1e-8)) > 0).astype(np.float32)
        rx_ls = (np.real(rx_symbols / (H_ls + 1e-8)) > 0).astype(np.float32)

        inp = np.stack([np.real(rx_symbols), np.imag(rx_symbols)], axis=1).flatten()
        inp = np.concatenate([inp, np.real(pilots_tx), np.imag(pilots_tx)])

        X_snr.append(inp)
        Y_snr.append(tx_bits.astype(np.float32))
        Y_mmse_snr.append(rx_mmse)
        Y_ls_snr.append(rx_ls)

    X_snr = np.array(X_snr)
    X_snr = scaler.transform(X_snr)
    Y_snr = np.array(Y_snr)
    Y_mmse_snr = np.array(Y_mmse_snr)
    Y_ls_snr = np.array(Y_ls_snr)
    Y_dnn_pred = model.predict(X_snr) > 0.5

    ber_dnn_curve.append(np.mean(Y_dnn_pred != Y_snr))
    ber_mmse_curve.append(np.mean(Y_mmse_snr != Y_snr))
    ber_ls_curve.append(np.mean(Y_ls_snr != Y_snr))

plt.figure()
plt.semilogy(SNR_range, ber_dnn_curve, 'o-', label='DNN')
plt.semilogy(SNR_range, ber_mmse_curve, 's-', label='MMSE')
plt.semilogy(SNR_range, ber_ls_curve, 'x-', label='LS')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which='both')
plt.legend()
plt.title("BER vs SNR")
plt.show()