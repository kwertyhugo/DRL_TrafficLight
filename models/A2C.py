import os
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Input, LayerNormalization
from keras.optimizers import Adam
from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class A2CAgent:
    
    def __init__(self, state_size, action_size, gamma=0.95, learning_rate=0.0003, 
                entropy_coef=0.2, value_coef=0.5, max_grad_norm=0.5, name='A2C_Agent'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.name = name
        
        # Episode trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Build model
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
        # Track training stats
        self.episode_count = 0
        
    def _build_model(self):
        """Build Actor-Critic model with normalization and regularization"""
        inputs = Input(shape=(self.state_size,))
        
        # Normalize input
        x = LayerNormalization()(inputs)
        
        # Shared layers with regularization
        x = Dense(64, activation='tanh', 
                 kernel_regularizer=regularizers.l2(0.01),
                 kernel_initializer='glorot_uniform')(x)
        x = LayerNormalization()(x)
        
        x = Dense(32, activation='tanh',
                 kernel_regularizer=regularizers.l2(0.01),
                 kernel_initializer='glorot_uniform')(x)
        x = LayerNormalization()(x)
        
        # Actor head (policy) - output log probabilities for numerical stability
        actor_hidden = Dense(16, activation='tanh',
                            kernel_initializer='glorot_uniform')(x)
        action_logits = Dense(self.action_size, 
                             kernel_initializer='glorot_uniform',
                             name='actor')(actor_hidden)
        
        # Critic head (value)
        critic_hidden = Dense(16, activation='tanh',
                             kernel_initializer='glorot_uniform')(x)
        state_value = Dense(1, 
                           kernel_initializer='glorot_uniform',
                           name='critic')(critic_hidden)
        
        model = Model(inputs=inputs, outputs=[action_logits, state_value])
        return model
    
    def act(self, state, training=True):
        """Choose action using softmax policy"""
        state = state.reshape((1, self.state_size))
        action_logits, state_value = self.model.predict(state, verbose=0)
        
        # Convert logits to probabilities
        action_probs = tf.nn.softmax(action_logits[0]).numpy()
        
        if training:
            # Sample from distribution
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            # Greedy action
            action = np.argmax(action_probs)
        
        return action
    
    def store_transition(self, state, action, reward):
        """Store transition in episode buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train_on_episode(self):
        """Train on collected episode with heavy stabilization"""
        if len(self.states) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        self.episode_count += 1
        
        # Convert to arrays
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        
        # Compute returns with normalization
        returns = self._compute_returns(rewards)
        
        # Normalize returns (critical for stability)
        if len(returns) > 1 and np.std(returns) > 1e-8:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Train
        actor_loss, critic_loss, entropy = self._train_step(states, actions, returns)
        
        total_reward = np.sum(rewards)
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        
        return actor_loss, critic_loss, entropy, total_reward
    
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    @tf.function
    def _train_step(self, states, actions, returns):
        """Training step with strong constraints"""
        with tf.GradientTape() as tape:
            # Forward pass
            action_logits, values = self.model(states, training=True)
            values = tf.squeeze(values)
            
            # Get action probabilities from logits
            action_probs = tf.nn.softmax(action_logits)
            
            # Compute advantages
            advantages = returns - values
            
            # Normalize advantages
            adv_mean = tf.reduce_mean(advantages)
            adv_std = tf.math.reduce_std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std
            
            # One-hot encode actions
            actions_onehot = tf.one_hot(actions, self.action_size)
            
            # Get log probabilities of taken actions
            selected_action_probs = tf.reduce_sum(action_probs * actions_onehot, axis=1)
            # Add epsilon and clip for numerical stability
            selected_action_probs = tf.clip_by_value(selected_action_probs, 1e-8, 1.0 - 1e-8)
            log_probs = tf.math.log(selected_action_probs)
            
            # Actor loss - MUST be positive (we minimize negative reward)
            # Stop gradient on advantages to prevent feedback loop
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
            
            # Force actor loss to be positive
            actor_loss = tf.abs(actor_loss)
            
            # Critic loss - MSE
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy for exploration - add epsilon for stability
            entropy = -tf.reduce_mean(
                tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1)
            )
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Compute gradients with clipping
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clip gradients aggressively
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return tf.abs(actor_loss), critic_loss, entropy
    
    def clear_trajectory(self):
        """Clear trajectory without training"""
        self.states = []
        self.actions = []
        self.rewards = []
    
    def load(self):
        """Load pre-trained model"""
        model_path = './agent_models/' + self.name + '.keras'
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = load_model(model_path)
        else:
            print(f"No model found at {model_path}, starting fresh.")
    
    def save(self):
        """Save model"""
        os.makedirs('./agent_models', exist_ok=True)
        self.model.save('./agent_models/' + self.name + '.keras')
        print(f"Model saved to ./agent_models/{self.name}.keras")