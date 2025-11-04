import os
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class A2CAgent:
    
    def __init__(self, state_size, action_size, gamma=0.95, learning_rate=0.0002, 
                entropy_coef=0.01, value_coef=0.5, name='A2C_Agent'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef  # For exploration
        self.value_coef = value_coef      # Weight for critic loss
        self.name = name
        
        # Store trajectory for training (on-policy)
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
    def _build_model(self):
        """Build Actor-Critic model with shared layers"""
        inputs = Input(shape=(self.state_size,))
        
        # Shared layers
        x = Dense(512, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # Actor head (policy network)
        actor = Dense(64, activation='relu')(x)
        action_probs = Dense(self.action_size, activation='softmax', name='actor')(actor)
        
        # Critic head (value network)
        critic = Dense(64, activation='relu')(x)
        state_value = Dense(1, activation='linear', name='critic')(critic)
        
        model = Model(inputs=inputs, outputs=[action_probs, state_value])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def act(self, state):
        """Choose action by sampling from policy distribution"""
        state = state.reshape((1, self.state_size))
        action_probs, state_value = self.model.predict(state, verbose=0)
        
        # Store value for later training
        self.values.append(state_value[0, 0])
        
        # Sample action from probability distribution
        action = np.random.choice(self.action_size, p=action_probs[0])
        return action
    
    def train(self):
        """Train on collected trajectory"""
        if len(self.states) == 0:
            return 0, 0, 0  # No data to train on
        
        # Convert lists to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards, dones)
        advantages = returns - values
        
        # Normalize advantages for stability
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Train the model
        actor_loss, critic_loss, entropy = self._train_step(states, actions, advantages, returns)
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return actor_loss, critic_loss, entropy
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns"""
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0
        
        # Calculate returns backwards
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            
        return returns
    
    @tf.function
    def _train_step(self, states, actions, advantages, returns):
        """Single training step using GradientTape"""
        with tf.GradientTape() as tape:
            # Forward pass
            action_probs, values = self.model(states, training=True)
            values = tf.squeeze(values)
            
            # Convert actions to one-hot
            actions_onehot = tf.one_hot(actions, self.action_size)
            
            # Get probabilities of taken actions
            action_log_probs = tf.math.log(tf.reduce_sum(action_probs * actions_onehot, axis=1) + 1e-10)
            
            # Actor loss (policy gradient)
            actor_loss = -tf.reduce_mean(action_log_probs * advantages)
            
            # Critic loss (value prediction error)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy bonus for exploration
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return actor_loss.numpy(), critic_loss.numpy(), entropy.numpy()
    
    def load(self):
        """Load a pre-trained model"""
        model_path = './agent_models/' + self.name + '.keras'
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
    
    def save(self):
        """Save the current model"""
        self.model.save('./agent_models/' + self.name + '.keras')