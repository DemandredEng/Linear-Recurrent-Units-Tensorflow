import tensorflow as tf
import math 
import tensorflow_probability as tfp
parallel_scan = tfp.math.scan_associative

class LRU(tf.keras.layers.Layer):
    def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.283):
        super(LRU, self).__init__()
        # Initialize layer parameters
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
        # Binary operator for parallel scan of linear recurrence.
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j
    
    def call(self, input_sequence):
    nu_log, theta_log, B, C, D, gamma_log = self.lru_parameters
    
    # Get dynamic dimensions - CHANGE 1: Add dynamic shape handling
    batch_size, seq_len = tf.shape(input_sequence)[0], tf.shape(input_sequence)[1]
    
    # Materializing the diagonal of Lambda and projections
    Lambda = tf.math.exp(tf.complex(-tf.math.exp(nu_log), tf.math.exp(theta_log)))
    exp_gamma_log = tf.math.exp(tf.complex(tf.zeros_like(gamma_log), gamma_log))
    B_norm = B * tf.expand_dims(exp_gamma_log, axis = -1)

    # Convert real input sequences to complex form if needed
    if input_sequence.dtype.is_complex:
        input_sequence_complex = input_sequence
    else:
        input_sequence_complex = tf.complex(input_sequence, tf.zeros_like(input_sequence))
    
    Bu_elements = tf.einsum('bth,nh->btn', input_sequence_complex, B_norm)
    Lambda_elements = tf.broadcast_to(Lambda[None, None, :], [seq_len, batch_size, self.N])
    Bu_elements_transposed = tf.transpose(Bu_elements, [1, 0, 2])
    elements = (Lambda_elements, Bu_elements_transposed)
    
    _, inner_states = parallel_scan(self.binary_operator_diag, elements)
    
    inner_states = tf.transpose(inner_states, [1, 0, 2])
    D = tf.cast(D, tf.complex64)
    y = tf.einsum('btn,hn->bth', inner_states, C) + tf.expand_dims(tf.expand_dims(D, 0), 0) * input_sequence_complex
    
    return tf.math.real(y)

