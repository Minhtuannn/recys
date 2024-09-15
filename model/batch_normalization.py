### batch_normol

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
# decay = 0.95
# TAU = 0.001

# class batch_norm:
#     def __init__(self,inputs,size,is_training,sess,parForTarget=None,bn_param=None):
        
#         self.sess = sess        
#         self.scale = tf.Variable(tf.random_uniform([size],0.9,1.1))
#         self.beta = tf.Variable(tf.random_uniform([size],-0.01,0.01))
#         self.pop_mean = tf.Variable(tf.random_uniform([size],-0.01,0.01),trainable=False)
#         self.pop_var = tf.Variable(tf.random_uniform([size],0.9,1.1),trainable=False)        
#         self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])        
#         self.train_mean = tf.assign(self.pop_mean,self.pop_mean * decay + self.batch_mean * (1 - decay))  
#         self.train_var = tf.assign(self.pop_var,self.pop_var * decay + self.batch_var * (1 - decay))
                
#         def training(): 
#             return tf.nn.batch_normalization(inputs,
#                 self.batch_mean, self.batch_var, self.beta, self.scale, 0.0000001 )
    
#         def testing(): 
#             return tf.nn.batch_normalization(inputs,
#             self.pop_mean, self.pop_var, self.beta, self.scale, 0.0000001)
        
#         if parForTarget!=None:
#             self.parForTarget = parForTarget
#             self.updateScale = self.scale.assign(self.scale*(1-TAU)+self.parForTarget.scale*TAU)
#             self.updateBeta = self.beta.assign(self.beta*(1-TAU)+self.parForTarget.beta*TAU)
#             self.updateTarget = tf.group(self.updateScale, self.updateBeta)
    
#         self.bnorm = tf.cond(is_training,training,testing)
    
#     def update_Target(self):
#         self.sess.run(self.updateBeta)
#         self.sess.run(self.updateScale)


import tensorflow as tf

decay = 0.95
TAU = 0.001

class batch_norm:
    def __init__(self, inputs, size, is_training, par_for_target=None):
        self.scale = tf.Variable(tf.random.uniform([size], 0.9, 1.1), trainable=True)
        self.beta = tf.Variable(tf.random.uniform([size], -0.01, 0.01), trainable=True)
        self.pop_mean = tf.Variable(tf.random.uniform([size], -0.01, 0.01), trainable=False)
        self.pop_var = tf.Variable(tf.random.uniform([size], 0.9, 1.1), trainable=False)

        # Compute batch statistics for the current batch
        self.batch_mean, self.batch_var = tf.nn.moments(inputs, [0])

        # Reshape batch_mean and batch_var to match pop_mean and pop_var shapes
        self.batch_mean = tf.squeeze(self.batch_mean)
        self.batch_var = tf.squeeze(self.batch_var)

        # Update population statistics (this should be done as an operation)
        def update_population():
            pop_mean_update = self.pop_mean.assign(self.pop_mean * decay + self.batch_mean * (1 - decay))
            pop_var_update = self.pop_var.assign(self.pop_var * decay + self.batch_var * (1 - decay))
            return pop_mean_update, pop_var_update

        # Select whether to use batch statistics or population statistics
        if is_training:
            # When training, use batch statistics
            self.bnorm = tf.nn.batch_normalization(inputs,
                                                   self.batch_mean, self.batch_var,
                                                   self.beta, self.scale, 1e-7)
            # Ensure population statistics are updated during training
            update_population()
        else:
            # When not training (inference), use population statistics
            self.bnorm = tf.nn.batch_normalization(inputs,
                                                   self.pop_mean, self.pop_var,
                                                   self.beta, self.scale, 1e-7)

        # Handle target network updates (for DDPG)
        if par_for_target is not None:
            self.update_scale = self.scale.assign(self.scale * (1 - TAU) + par_for_target.scale * TAU)
            self.update_beta = self.beta.assign(self.beta * (1 - TAU) + par_for_target.beta * TAU)
            self.update_target = tf.group(self.update_scale, self.update_beta)

    def update_population_stats(self):
        # This method updates the population statistics (mean and variance)
        self.pop_mean.assign(self.batch_mean)
        self.pop_var.assign(self.batch_var)

    def update_target_network(self):
        # Update the target network parameters (used in DDPG target network soft updates)
        self.update_scale.assign(self.scale * (1 - TAU) + self.par_for_target.scale * TAU)
        self.update_beta.assign(self.beta * (1 - TAU) + self.par_for_target.beta * TAU)


