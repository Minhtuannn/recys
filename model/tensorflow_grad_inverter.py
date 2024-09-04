## tensor grad_inverted

import tensorflow as tf

class GradInverter:
    def __init__(self, action_bounds):
        self.action_size = len(action_bounds[0])
        self.pmax = tf.constant(action_bounds[0], dtype=tf.float32)
        self.pmin = tf.constant(action_bounds[1], dtype=tf.float32)
        self.prange = tf.constant([x - y for x, y in zip(action_bounds[0], action_bounds[1])], dtype=tf.float32)

    def invert(self, grad, action):
        # Ensure inputs are tensors
        action_input = tf.convert_to_tensor(action, dtype=tf.float32)
        act_grad = tf.convert_to_tensor(grad, dtype=tf.float32)

        pdiff_max = (-action_input + self.pmax) / self.prange
        pdiff_min = (action_input - self.pmin) / self.prange
        zeros_act_grad_filter = tf.zeros_like(act_grad)

        # Perform element-wise comparison and selection
        grad_inverter = tf.where(tf.greater(act_grad, zeros_act_grad_filter), act_grad * pdiff_max, act_grad * pdiff_min)
        
        return grad_inverter.numpy()  # Convert the result back to numpy if necessary
