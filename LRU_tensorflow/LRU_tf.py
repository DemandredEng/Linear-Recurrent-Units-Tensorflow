import tensorflow as tf
import math 
import tensorflow_probability as tfp
parallel_scan = tfp.math.scan_associative

class LRU(tf.keras.layers.Layer):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.283):
        super(LRU, self).__init__()
        self.N = N
        self.H = H
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.lru_parameters = self.init_lru_parameters()

    def init_lru_parameters(self):
        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring between r_min and r_max, with phase in [0, max_phase].
        u1 = tf.random.uniform(shape = (self.N,))
        u2 = tf.random.uniform(shape = (self.N,))
        nu_log = tf.math.log(-0.5 * tf.math.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        theta_log = tf.math.log(u2 * self.max_phase)

        # Glorot initialized Input/Output projection matrices
        B = tf.complex(tf.random.normal(shape = (self.N,self.H)) / math.sqrt(2*self.H), tf.random.normal(shape = (self.N,self.H)) / math.sqrt(2*self.H))
        C = tf.complex(tf.random.normal(shape = (self.H,self.N)) / math.sqrt(self.N), tf.random.normal(shape = (self.H,self.N)) / math.sqrt(self.N))
        D = tf.random.normal(shape = (self.H,))

        # Normalization factor
        diag_lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        gamma_log = tf.math.log(tf.math.sqrt(1 - tf.math.abs(diag_lambda)**2))

        return nu_log, theta_log, B, C, D, gamma_log
    
    def binary_operator_diag(self, element_i, element_j):
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j  

    def call(self, input_sequence):
        nu_log, theta_log, B, C, D, gamma_log = self.lru_parameters
        # Materializing the diagonal of Lambda and projections
        Lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
        exp_gamma_log = tf.math.exp(tf.complex(tf.zeros_like(gamma_log), gamma_log))
        B_norm = B * tf.expand_dims(exp_gamma_log, axis = -1)

        # Running the LRU + output projection
        Lambda_reshaped = tf.expand_dims(Lambda, axis=0)
        Lambda_elements = tf.repeat(Lambda_reshaped, repeats = input_sequence.shape[0], axis = 0)


        input_sequence_reshaped = tf.expand_dims(input_sequence, axis=-1)
        Bu_elements = tf.vectorized_map(lambda u: tf.linalg.matmul(B_norm, u), input_sequence_reshaped)
        Bu_elements = tf.squeeze(Bu_elements, axis=-1)

        elements = (Lambda_elements, Bu_elements)
        _, inner_states = parallel_scan(self.binary_operator_diag, elements)
        D = tf.cast(D, tf.complex64)
        y = tf.vectorized_map(lambda args: tf.math.real(tf.linalg.matvec(C, args[0])) + tf.math.real(D * args[1]), (inner_states, input_sequence))
        return y
        

N = 5  # State dimension
H = 3  # Model dimension
L = 10  # Number of time steps

real_parts = tf.random.uniform(shape=(L, H), dtype=tf.float32)
imaginary_parts = tf.random.uniform(shape=(L, H), dtype=tf.float32)
input_sequence = tf.complex(real_parts, imaginary_parts)

lru = LRU(N, H)
preds = lru(input_sequence)

print(preds)
