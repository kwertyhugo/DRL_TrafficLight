import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# GPU memory growth
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_actor(state_size, action_size, name='Actor'):
    """Build actor network that outputs action probabilities."""
    state_input = Input(shape=(state_size,), name='state_input')
    x = Dense(512, activation='relu')(state_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # Softmax for discrete action probabilities
    action_probs = Dense(action_size, activation='softmax', name='action_probs')(x)
    model = Model(inputs=state_input, outputs=action_probs, name=name)
    return model


def build_critic(state_size, name='Critic'):
    """Build critic network that outputs state value V(s)."""
    state_input = Input(shape=(state_size,), name='state_input')
    x = Dense(512, activation='relu')(state_input)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # Single output for state value
    value = Dense(1, activation='linear', name='value')(x)
    model = Model(inputs=state_input, outputs=value, name=name)
    return model


class A2CAgent:
    def __init__(
        self,
        state_size,
        action_size,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        name='A2CAgent'
    ):
        """
        A2C Agent for discrete action spaces.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor
            entropy_coef: Coefficient for entropy bonus (encourages exploration)
            value_coef: Coefficient for value loss
            name: Agent name for saving
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.name = name

        # Build networks
        self.actor = build_actor(state_size, action_size, name='actor')
        self.critic = build_critic(state_size, name='critic')

        # Optimizers
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

        # Storage for episode trajectory
        self.states = []
        self.actions = []
        self.rewards = []

    def get_action(self, state, training=True):
        """
        Sample action from policy distribution.
        
        Args:
            state: Current state
            training: If True, sample stochastically; if False, take argmax
        
        Returns:
            action: Selected action index
        """
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        action_probs = self.actor.predict(state, verbose=0)[0]

        if training:
            # Sample from probability distribution
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            # Take most likely action (for evaluation)
            action = np.argmax(action_probs)

        return action

    def remember(self, state, action, reward):
        """Store transition in episode buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        """Clear episode buffer."""
        self.states = []
        self.actions = []
        self.rewards = []

    def compute_returns(self, next_state_value, done):
        """
        Compute discounted returns for the episode.
        
        Args:
            next_state_value: Value of the final next state (0 if terminal)
            done: Whether episode is done
        
        Returns:
            returns: List of discounted returns for each step
        """
        returns = []
        R = 0.0 if done else next_state_value

        # Compute returns backwards from the end of episode
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return np.array(returns, dtype=np.float32)

    @tf.function
    def _train_step(self, states, actions, returns):
        """
        Single training step for both actor and critic.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            returns: Batch of discounted returns
        
        Returns:
            actor_loss, critic_loss, entropy
        """
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Forward pass
            action_probs = self.actor(states, training=True)
            state_values = self.critic(states, training=True)
            state_values = tf.squeeze(state_values, axis=1)

            # Compute advantages
            advantages = returns - state_values

            # Critic loss (MSE between predicted value and actual return)
            critic_loss = tf.reduce_mean(tf.square(advantages))

            # Actor loss (policy gradient with advantage)
            actions_one_hot = tf.one_hot(actions, self.action_size)
            log_probs = tf.math.log(tf.reduce_sum(action_probs * actions_one_hot, axis=1) + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

            # Entropy bonus (encourages exploration)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            )

            # Total actor loss with entropy bonus
            total_actor_loss = actor_loss - self.entropy_coef * entropy

            # Total critic loss with coefficient
            total_critic_loss = self.value_coef * critic_loss

        # Compute and apply gradients
        actor_grads = actor_tape.gradient(total_actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(total_critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss, critic_loss, entropy

    def train(self, next_state, done):
        """
        Train on collected episode data.
        
        Args:
            next_state: The final next state of the episode
            done: Whether the episode terminated
        
        Returns:
            actor_loss, critic_loss, entropy (all as floats)
        """
        if len(self.states) == 0:
            return None, None, None

        # Prepare data
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int32)

        # Compute value of next state for bootstrapping
        next_state_value = 0.0
        if not done:
            next_state = np.asarray(next_state, dtype=np.float32).reshape(1, -1)
            next_state_value = float(self.critic.predict(next_state, verbose=0)[0, 0])

        # Compute returns
        returns = self.compute_returns(next_state_value, done)

        # Train
        actor_loss, critic_loss, entropy = self._train_step(states, actions, returns)

        # Clear episode memory
        self.clear_memory()

        return float(actor_loss.numpy()), float(critic_loss.numpy()), float(entropy.numpy())

    def save(self, folder='agent_weights'):
        """Save model weights."""
        os.makedirs(folder, exist_ok=True)
        self.actor.save_weights(os.path.join(folder, f'{self.name}_actor.h5'))
        self.critic.save_weights(os.path.join(folder, f'{self.name}_critic.h5'))

    def load(self, folder='agent_weights'):
        """Load model weights."""
        self.actor.load_weights(os.path.join(folder, f'{self.name}_actor.h5'))
        self.critic.load_weights(os.path.join(folder, f'{self.name}_critic.h5'))