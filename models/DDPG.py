import os
import random
from collections import deque
import numpy as np
import pickle
import csv
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, GRU, LayerNormalization
from keras.optimizers import Adam
from keras.models import load_model

# GPU memory growth (same pattern you used)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class OUActionNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
    def __init__(self, mean, std_deviation=0.3, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = np.array(mean, dtype=np.float32)
        self.std_dev = std_deviation
        self.dt = dt
        self.x_prev = x_initial if x_initial is not None else np.zeros_like(self.mean, dtype=np.float32)

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        ).astype(np.float32)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mean, dtype=np.float32)


class ReplayBuffer:
    def __init__(self, maxlen=1000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        states = np.asarray([e[0] for e in batch], dtype=np.float32)
        actions = np.asarray([e[1] for e in batch], dtype=np.float32)
        rewards = np.asarray([e[2] for e in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.asarray([e[3] for e in batch], dtype=np.float32)
        dones = np.asarray([e[4] for e in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def build_actor(state_size, action_size, action_low, action_high, name='Actor'):
    state_input = Input(shape=(state_size,), name='state_input')
    
    # Increased layer width to handle consolidated state information
    x = Dense(512, activation='relu')(state_input)
    x = LayerNormalization()(x) # Added normalization for stability in continuous space
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    raw_actions = Dense(action_size, activation='tanh')(x)  # Outputs vector of size 3 (-1..1)
    
    # Scaling logic remains the same, but it now applies to a vector of 3
    scale = (action_high - action_low) / 2.0
    mean = (action_high + action_low) / 2.0
    out = tf.keras.layers.Lambda(lambda a: a * scale + mean, name='scaled_actions')(raw_actions)
    
    model = Model(inputs=state_input, outputs=out, name=name)
    return model


def build_critic(state_size, action_size, name='Critic'):
    state_input = Input(shape=(state_size,), name='state_input')
    action_input = Input(shape=(action_size,), name='action_input')
    
    # State branch
    xs = Dense(512, activation='relu')(state_input)
    xs = LayerNormalization()(xs)
    xs = Dense(256, activation='relu')(xs)
    
    # Action branch
    xa = Dense(256, activation='relu')(action_input)
    
    # Combine - The critic sees all 11 state features and all 3 actions
    x = Concatenate()([xs, xa])
    x = Dense(512, activation='relu')(x) # Increased depth
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    q_out = Dense(1, activation='linear')(x)
    model = Model(inputs=[state_input, action_input], outputs=q_out, name=name)
    return model


class DDPGAgent:
    def __init__(
        self,
        state_size,
        action_size,
        action_low,
        action_high,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        name='DDPGAgent'
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = np.array(action_low, dtype=np.float32)
        self.action_high = np.array(action_high, dtype=np.float32)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.name = name

        # Replay buffer
        self.replay_buffer = ReplayBuffer(maxlen=buffer_size)

        # Actor & Critic networks
        self.actor = build_actor(state_size, action_size, self.action_low, self.action_high, name='actor')
        self.critic = build_critic(state_size, action_size, name='critic')

        # Target networks
        self.target_actor = build_actor(state_size, action_size, self.action_low, self.action_high, name='target_actor')
        self.target_critic = build_critic(state_size, action_size, name='target_critic')

        # Optimizers
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

        # Initialize target weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Exploration noise
        self.noise = OUActionNoise(mean=np.zeros(self.action_size), std_deviation=0.2)

    def get_action(self, state, add_noise=True):
        """Return a single action for given state (1D state)."""
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        action = self.actor.predict(state, verbose=0)[0]
        if add_noise:
            action = action + self.noise()
        # clip to bounds
        return np.clip(action, self.action_low, self.action_high)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(self.tau * b + (1.0 - self.tau) * a)

    @tf.function
    def _critic_train_step(self, states, actions, y):
        """Train critic (single TF function for performance)."""
        with tf.GradientTape() as tape:
            q_values = self.critic([states, actions], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - q_values))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        return critic_loss

    @tf.function
    def _actor_train_step(self, states):
        """Train actor (single TF function)."""
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_value = self.critic([states, actions], training=True)
            # Want to maximize Q, minimize -Q
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        return actor_loss

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None  # not enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute target Q values:
        target_actions = self.target_actor.predict(next_states, verbose=0)
        target_qs = self.target_critic.predict([next_states, target_actions], verbose=0)
        y = rewards + self.gamma * (1 - dones) * target_qs

        # Train critic
        critic_loss = self._critic_train_step(states, actions, y)

        # Train actor
        actor_loss = self._actor_train_step(states)

        # Soft update target networks
        self.update_target(self.target_actor.variables, self.actor.variables)
        self.update_target(self.target_critic.variables, self.critic.variables)

        return float(actor_loss.numpy()), float(critic_loss.numpy())

    def save(self, folder='./Olivarez_traci/models_DDPG/'):
        os.makedirs(folder, exist_ok=True)
        self.actor.save_weights(os.path.join(folder, f'{self.name}_actor_weights.weights.h5'))
        self.critic.save_weights(os.path.join(folder, f'{self.name}_critic_weights.weights.h5'))
        self.target_actor.save_weights(os.path.join(folder, f'{self.name}_target_actor_weights.weights.h5'))
        self.target_critic.save_weights(os.path.join(folder, f'{self.name}_target_critic_weights.weights.h5'))
        print("All weights saved successfully.")

    def load(self, folder='./Olivarez_traci/models_DDPG/'):
        actor_w = os.path.join(folder, f'{self.name}_actor_weights.weights.h5')
        critic_w = os.path.join(folder, f'{self.name}_critic_weights.weights.h5')
        target_actor_w = os.path.join(folder, f'{self.name}_target_actor_weights.weights.h5')
        target_critic_w = os.path.join(folder, f'{self.name}_target_critic_weights.weights.h5')

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)
        print(f"Loading model weights from '{folder}'...")

        # Check for missing files
        missing = [p for p in [actor_w, critic_w, target_actor_w, target_critic_w] if not os.path.exists(p)]

        if missing:
            print("Warning: Some weight files are missing:")
            for f in missing:
                print(f"  - {f}")
            print("Creating new placeholder weights...")
            # Initialize and save empty weights
            try:
                self.actor.save_weights(actor_w)
                self.critic.save_weights(critic_w)
                self.target_actor.save_weights(target_actor_w)
                self.target_critic.save_weights(target_critic_w)
                print("Placeholder weight files created successfully.")
            except Exception as e:
                print("Error while creating placeholder weights:", e)
            return

        # If all weights exist, load them
        try:
            self.actor.load_weights(actor_w)
            self.critic.load_weights(critic_w)
            self.target_actor.load_weights(target_actor_w)
            self.target_critic.load_weights(target_critic_w)
            print("All weights loaded successfully.")
        except Exception as e:
            print("Error while loading weights:", e)
            
    def save_replay_buffer(self, folder='./Olivarez_traci/models_DDPG/'):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{self.name}_replay.pkl')
        try:
            with open(path, 'wb') as f:
                pickle.dump(list(self.replay_buffer.buffer), f)
            print(f"[INFO] Replay buffer saved for {self.name} -> {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save replay buffer for {self.name}: {e}")

    def load_replay_buffer(self, folder='./Olivarez_traci/models_DDPG/'):
        path = os.path.join(folder, f'{self.name}_replay.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.replay_buffer.buffer = deque(data, maxlen=self.replay_buffer.buffer.maxlen)
                print(f"[INFO] Replay buffer loaded for {self.name} ({len(self.replay_buffer)} experiences)")
            except Exception as e:
                print(f"[ERROR] Failed to load replay buffer for {self.name}: {e}")
        else:
            print(f"[WARN] No replay buffer found for {self.name} at {path}")

                     
    def load_history(self, filepath):
        rewards, actor_losses, critic_losses = [], [], []
        if not os.path.exists(filepath):
            print(f"[WARN] No training history file found at {filepath}")
            return rewards, actor_losses, critic_losses
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rewards.append(float(row['Reward']))
                    actor_losses.append(float(row['Actor_Loss']))
                    critic_losses.append(float(row['Critic_Loss']))
            print(f"[INFO] Loaded training history from {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to load training history from {filepath}: {e}")
        return rewards, actor_losses, critic_losses
    
    def decay_noise(self, decay_rate=0.995, min_std=0.05):
        """Decay OU noise standard deviation gradually to stabilize training."""
        old_std = self.noise.std_dev
        self.noise.std_dev = max(min_std, self.noise.std_dev * decay_rate)
        print(f"[NOISE] {self.name}: Noise std decayed from {old_std:.4f} → {self.noise.std_dev:.4f}")




