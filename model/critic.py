import tensorflow as tf
import math
import numpy as np

LEARNING_RATE = 0.001
TAU = 0.001
BATCH_SIZE = 64
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300

class CriticNetwork(tf.Module):
    """Critic Q value model backbone with batch normalization for DDPG"""
    
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Critic Network weights and biases
        self.W1_c = tf.Variable(tf.random.uniform([num_states, N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
        self.B1_c = tf.Variable(tf.random.uniform([N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
        
        self.W2_c = tf.Variable(tf.random.uniform([N_HIDDEN_1, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1 + num_actions), 1/math.sqrt(N_HIDDEN_1 + num_actions)))
        self.B2_c = tf.Variable(tf.random.uniform([N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1 + num_actions), 1/math.sqrt(N_HIDDEN_1 + num_actions)))
        
        self.W2_action_c = tf.Variable(tf.random.uniform([num_actions, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1 + num_actions), 1/math.sqrt(N_HIDDEN_1 + num_actions)))
        
        self.W3_c = tf.Variable(tf.random.uniform([N_HIDDEN_2, 1], -0.003, 0.003))
        self.B3_c = tf.Variable(tf.random.uniform([1], -0.003, 0.003))
        
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
    
    def __call__(self, state, action, training):
        h1_t = tf.matmul(state, self.W1_c)
        h1_c_bn = self.batch_norm1(h1_t, training=training)
        h1_c = tf.nn.softplus(h1_c_bn) + self.B1_c
        
        h2_t = tf.matmul(h1_c, self.W2_c) + tf.matmul(action, self.W2_action_c)
        h2_c_bn = self.batch_norm2(h2_t, training=training)
        h2_c = tf.nn.tanh(h2_c_bn) + self.B2_c
        
        q_value = tf.matmul(h2_c, self.W3_c) + self.B3_c
        return q_value
    
    def get_variables(self):
        return [self.W1_c, self.B1_c, self.W2_c, self.B2_c, self.W2_action_c, self.W3_c, self.B3_c]


class Critic:
    """Critic that handles network training, target update, and other tasks"""
    
    def __init__(self, num_states, num_actions):
        # Initialize main and target networks
        self.critic_network = CriticNetwork(num_states, num_actions)
        self.target_network = CriticNetwork(num_states, num_actions)
        
        # Initialize target network with same weights as main network
        self.update_target_critic(initial=True)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def train_critic(self, state_batch, action_batch, y_batch):
        with tf.GradientTape() as tape:
            q_values = self.critic_network(state_batch, action_batch, training=True)
            l2_loss = 0.0001 * tf.reduce_sum(tf.square(self.critic_network.W2_c))
            cost = tf.reduce_mean(tf.square(q_values - y_batch)) + l2_loss
        
        critic_variables = self.critic_network.get_variables()
        gradients = tape.gradient(cost, critic_variables)
        self.optimizer.apply_gradients(zip(gradients, critic_variables))

    def evaluate_target_critic(self, state_batch, action_batch):
        return self.target_network(state_batch, action_batch, training=False)
    
    def compute_delQ_a(self, state_batch, action_batch):
        with tf.GradientTape() as tape:
            tape.watch(action_batch)
            q_values = self.critic_network(state_batch, action_batch, training=False)
        return tape.gradient(q_values, action_batch)
    
    def update_target_critic(self, initial=False):
        if initial:
            for target_var, var in zip(self.target_network.get_variables(), self.critic_network.get_variables()):
                target_var.assign(var)
        else:
            for target_var, var in zip(self.target_network.get_variables(), self.critic_network.get_variables()):
                target_var.assign(TAU * var + (1 - TAU) * target_var)
