# import tensorflow as tf
# import numpy as np
# import math

# from model.batch_normalization import batch_norm
# LEARNING_RATE = 0.0001
# TAU = 0.001
# N_HIDDEN_1 = 400
# N_HIDDEN_2 = 300

# class ActorNetwork(tf.Module):
#     """Backbone Network for the Actor"""
    
#     def __init__(self, num_states, num_actions):
#         self.num_states = num_states
#         self.num_actions = num_actions
        
#         # Network parameters
#         self.W1 = tf.Variable(tf.random.uniform([num_states, N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
#         self.B1 = tf.Variable(tf.random.uniform([N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
#         self.W2 = tf.Variable(tf.random.uniform([N_HIDDEN_1, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1)))
#         self.B2 = tf.Variable(tf.random.uniform([N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1)))
#         self.W3 = tf.Variable(tf.random.uniform([N_HIDDEN_2, num_actions], -0.003, 0.003))
#         self.B3 = tf.Variable(tf.random.uniform([num_actions], -0.003, 0.003))
        
#         # self.batch_norm1 =  batch_norm() #tf.keras.layers.BatchNormalization()
#         # self.batch_norm2 = batch_norm() #tf.keras.layers.BatchNormalization()
    
#     def __call__(self, state, training):
#         h1_t = tf.matmul(state, self.W1)
#         h1_bn = batch_norm(h1_t,N_HIDDEN_1, is_training=training)
#         h1 = tf.nn.softplus(h1_bn) + self.B1
        
#         h2_t = tf.matmul(h1, self.W2)
#         h2_bn = batch_norm(h2_t,N_HIDDEN_2, is_training=training)
#         h2 = tf.nn.tanh(h2_bn) + self.B2
        
#         return tf.matmul(h2, self.W3) + self.B3
    
#     def get_variables(self):
#         return [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]

# class Actor:
#     """Actor that handles the network training and target update"""
    
#     def __init__(self, num_states, num_actions):
#         # Initialize main and target networks
#         self.actor_network = ActorNetwork(num_states, num_actions)
#         self.target_network = ActorNetwork(num_states, num_actions)
        
#         # Initialize target network with same weights as main network
#         self.update_target_actor(initial=True)
        
#         # Optimizer
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)

#     def evaluate_actor(self, state):
#         return self.actor_network(state, training=False)
    
#     def evaluate_target_actor(self, state):
#         return self.target_network(state, training=False)
    
#     def train_actor(self, actor_state_in, q_gradient_input):
#         with tf.GradientTape() as tape:
#             actions = self.actor_network(actor_state_in, training=True)
#             actor_parameters = self.actor_network.get_variables()
#             gradients = tape.gradient(actions, actor_parameters, output_gradients=-q_gradient_input)
        
#         self.optimizer.apply_gradients(zip(gradients, actor_parameters))
    
#     def update_target_actor(self, initial=False):
#         if initial:
#             for target_var, var in zip(self.target_network.get_variables(), self.actor_network.get_variables()):
#                 target_var.assign(var)
#         else:
#             for target_var, var in zip(self.target_network.get_variables(), self.actor_network.get_variables()):
#                 target_var.assign(TAU * var + (1 - TAU) * target_var)

import tensorflow as tf
import math
from model.batch_normalization import batch_norm

# Constants
LEARNING_RATE = 0.0001
TAU = 0.001
N_HIDDEN_1 = 400
N_HIDDEN_2 = 300

class ActorNetwork(tf.Module):
    """Backbone Network for the Actor"""
    
    def __init__(self, num_states, num_actions):
        super(ActorNetwork, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Network parameters
        self.W1 = tf.Variable(tf.random.uniform([num_states, N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
        self.B1 = tf.Variable(tf.random.uniform([N_HIDDEN_1], -1/math.sqrt(num_states), 1/math.sqrt(num_states)))
        
        self.W2 = tf.Variable(tf.random.uniform([N_HIDDEN_1, N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1)))
        self.B2 = tf.Variable(tf.random.uniform([N_HIDDEN_2], -1/math.sqrt(N_HIDDEN_1), 1/math.sqrt(N_HIDDEN_1)))
        
        self.W3 = tf.Variable(tf.random.uniform([N_HIDDEN_2, num_actions], -0.003, 0.003))
        self.B3 = tf.Variable(tf.random.uniform([num_actions], -0.003, 0.003))
        
    def __call__(self, state, training):
        # First hidden layer with batch normalization and activation
        h1_t = tf.matmul(state, self.W1) + self.B1
        h1_bn = batch_norm(h1_t, N_HIDDEN_1, is_training=training).bnorm
        h1 = tf.nn.softplus(h1_bn)
        
        # Second hidden layer with batch normalization and activation
        h2_t = tf.matmul(h1, self.W2) + self.B2
        h2_bn = batch_norm(h2_t, N_HIDDEN_2, is_training=training).bnorm
        h2 = tf.nn.tanh(h2_bn)
        
        # Output layer
        return tf.matmul(h2, self.W3) + self.B3
    
    def get_variables(self):
        return [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]

class Actor:
    """Actor that handles the network training and target update"""
    
    def __init__(self, num_states, num_actions):
        # Initialize main and target networks
        self.actor_network = ActorNetwork(num_states, num_actions)
        self.target_network = ActorNetwork(num_states, num_actions)
        
        # Initialize target network with same weights as main network
        self.update_target_actor(initial=True)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)

    def evaluate_actor(self, state):
        """Evaluate the actor network for a given state (used in exploitation)"""
        return self.actor_network(state, training=False)
    
    def evaluate_target_actor(self, state):
        """Evaluate the target actor network for a given state"""
        return self.target_network(state, training=False)
    
    def train_actor(self, actor_state_in, q_gradient_input):
        """Train the actor network with Q-value gradients"""
        with tf.GradientTape() as tape:
            actions = self.actor_network(actor_state_in, training=True)
            actor_parameters = self.actor_network.get_variables()
            # The gradients for the actor are computed based on Q-value gradients from the critic
            gradients = tape.gradient(actions, actor_parameters, output_gradients=-q_gradient_input)
        
        # Apply the computed gradients to update the actor's parameters
        self.optimizer.apply_gradients(zip(gradients, actor_parameters))
    
    def update_target_actor(self, initial=False):
        """Update the target actor network with a soft update"""
        if initial:
            # Hard copy of weights from actor to target (initialization)
            for target_var, var in zip(self.target_network.get_variables(), self.actor_network.get_variables()):
                target_var.assign(var)
        else:
            # Soft update: TAU * actor + (1 - TAU) * target
            for target_var, var in zip(self.target_network.get_variables(), self.actor_network.get_variables()):
                target_var.assign(TAU * var + (1 - TAU) * target_var)
