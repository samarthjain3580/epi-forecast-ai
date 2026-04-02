import numpy as np
import os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTMPredictor:
    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_weights.npz")

        data = np.load(weights_path)
        self.kernel = data["lstm_kernel"]
        self.recurrent_kernel = data["lstm_recurrent_kernel"]
        self.lstm_bias = data["lstm_bias"]
        self.dense_kernel = data["dense_kernel"]
        self.dense_bias = data["dense_bias"]

        self.units = self.dense_kernel.shape[0]

        # Split into 4 gates: [input, forget, candidate, output]
        self.k_i = self.kernel[:, 0*self.units:1*self.units]
        self.k_f = self.kernel[:, 1*self.units:2*self.units]
        self.k_c = self.kernel[:, 2*self.units:3*self.units]
        self.k_o = self.kernel[:, 3*self.units:4*self.units]

        self.r_i = self.recurrent_kernel[:, 0*self.units:1*self.units]
        self.r_f = self.recurrent_kernel[:, 1*self.units:2*self.units]
        self.r_c = self.recurrent_kernel[:, 2*self.units:3*self.units]
        self.r_o = self.recurrent_kernel[:, 3*self.units:4*self.units]

        self.b_i = self.lstm_bias[0*self.units:1*self.units]
        self.b_f = self.lstm_bias[1*self.units:2*self.units]
        self.b_c = self.lstm_bias[2*self.units:3*self.units]
        self.b_o = self.lstm_bias[3*self.units:4*self.units]

    def predict(self, x_seq):
        x_seq = np.array(x_seq, dtype=np.float32)
        if x_seq.ndim == 2:
            x_seq = x_seq.reshape(1, *x_seq.shape)

        batch_size = x_seq.shape[0]
        h = np.zeros((batch_size, self.units), dtype=np.float32)
        c = np.zeros((batch_size, self.units), dtype=np.float32)

        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]

            i = sigmoid(x_t @ self.k_i + h @ self.r_i + self.b_i)
            f = sigmoid(x_t @ self.k_f + h @ self.r_f + self.b_f)
            c_candidate = np.tanh(x_t @ self.k_c + h @ self.r_c + self.b_c)
            o = sigmoid(x_t @ self.k_o + h @ self.r_o + self.b_o)

            c = f * c + i * c_candidate
            h = o * np.tanh(c)

        return h @ self.dense_kernel + self.dense_bias
