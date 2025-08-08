import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from chess_env import ChessEnv

# Hyperparameters
gamma = 0.99            # Discount factor for future rewards
epsilon_start = 1.0     # Initial exploration rate
epsilon_min = 0.1       # Minimum exploration rate
epsilon_decay = 0.995   # Multiplicative decay for epsilon per episode (or per step)
batch_size = 64         # Mini-batch size for training
memory_size = 20000     # Replay memory capacity
learning_rate = 0.001   # Learning rate for optimizer
target_update_interval = 50  # how often to update target network (episodes)

# Initialize the environment
env = ChessEnv()
state_size = env.observation_space.shape[0]   # 65
action_size = env.action_space.n              # 4096

def build_dqn_model(hidden_layers):
    """Build a DQN model with given hidden layer sizes."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(state_size,)))
    # Hidden layers with ReLU activations
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation='relu'))
    # Output layer: one Q-value per action
    model.add(keras.layers.Dense(action_size, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Training configurations for each difficulty
difficulties = {
    "easy":   {"episodes": 500,  "hidden_layers": [64, 64]},
    "medium": {"episodes": 2000, "hidden_layers": [128, 128]},
    "hard":   {"episodes": 5000, "hidden_layers": [256, 256, 256]}
}

# Dictionary to hold trained models (if we want to use them in code later)
trained_models = {}

for level, config in difficulties.items():
    episodes = config["episodes"]
    hidden_layers = config["hidden_layers"]
    print(f"Training {level.capitalize()} DQN for {episodes} episodes with network layers {hidden_layers}...")
    
    # Build Q-network and target network
    model = build_dqn_model(hidden_layers)
    target_model = build_dqn_model(hidden_layers)
    target_model.set_weights(model.get_weights())  # initialize target with same weights
    
    # Replay memory
    memory = deque(maxlen=memory_size)
    epsilon = epsilon_start
    
    for ep in range(1, episodes+1):
        state = env.reset()
        state = state.reshape(1, -1)  # reshape to 1x65 for network input
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            step_count += 1
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                # Exploration: random legal action
                # We will choose a random move from legal moves to avoid pointless illegal moves
                valid_moves = env.gs.get_valid_moves()
                if valid_moves:
                    move = random.choice(valid_moves)
                    from_idx = move.start_row * 8 + move.start_col
                    to_idx = move.end_row * 8 + move.end_col
                    action = from_idx * 64 + to_idx
                else:
                    action = np.random.randint(action_size)
            else:
                # Exploitation: choose best action from current Q-network
                q_values = model.predict(state, verbose=0)[0]  # shape (4096,)
                # Mask out illegal moves by setting them to a very low value
                valid_moves = env.gs.get_valid_moves()
                if valid_moves:
                    legal_action_indices = set(m.start_row*512 + m.start_col*64 + m.end_row*8 + m.end_col  # incorrect formula? Actually, need to re-calc
                        for m in valid_moves)
                else:
                    legal_action_indices = set()
                # Actually, the move encoding is from_idx*64 + to_idx; let's compute legal indices properly:
                legal_action_indices = set()
                for m in valid_moves:
                    start_index = m.start_row * 8 + m.start_col
                    end_index = m.end_row * 8 + m.end_col
                    legal_action_indices.add(start_index * 64 + end_index)
                if legal_action_indices:
                    # set Q-values for illegal actions to -inf so they won't be selected
                    # (We copy the array to not modify the original model output permanently)
                    q_values_masked = q_values.copy()
                    for a_idx in range(action_size):
                        if a_idx not in legal_action_indices:
                            q_values_masked[a_idx] = -1e9
                    action = int(np.argmax(q_values_masked))
                else:
                    # No legal moves (game over situation, but loop would break anyway)
                    action = int(np.argmax(q_values))
            # Take the selected action
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, -1)
            total_reward += reward
            # Store transition in replay memory
            memory.append((state, action, reward, next_state, done))
            # Update current state
            state = next_state
            
            # Train the Q-network on a batch from memory
            if len(memory) >= batch_size:
                # Sample a random batch
                batch = random.sample(memory, batch_size)
                states_mb = np.vstack([x[0] for x in batch])        # shape (batch_size, 65)
                actions_mb = np.array([x[1] for x in batch])
                rewards_mb = np.array([x[2] for x in batch])
                next_states_mb = np.vstack([x[3] for x in batch])
                dones_mb = np.array([x[4] for x in batch], dtype=bool)
                
                # Predict Q-values for current states and next states
                q_current = model.predict(states_mb, verbose=0)         # shape (batch_size, 4096)
                q_next_target = target_model.predict(next_states_mb, verbose=0)  # target network for stability
                
                # Prepare target Q values for training
                for i in range(batch_size):
                    action_idx = actions_mb[i]
                    if dones_mb[i]:
                        # If done, target is just the immediate reward
                        q_current[i, action_idx] = rewards_mb[i]
                    else:
                        # If not done, target = reward + gamma * max(Q_target(next_state))
                        max_future_q = np.max(q_next_target[i])
                        q_current[i, action_idx] = rewards_mb[i] + gamma * max_future_q
                # Train the main network toward the target Q values
                model.fit(states_mb, q_current, epochs=1, verbose=0)
            
            # If the game is over, break out of the loop
            if done:
                break
        
        # Decay exploration rate after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            if epsilon < epsilon_min:
                epsilon = epsilon_min
        
        # Update target network periodically
        if ep % target_update_interval == 0:
            target_model.set_weights(model.get_weights())
        
        # (Optional) Print episode stats for monitoring
        if ep % 10 == 0 or ep == episodes:
            print(f"Episode {ep}/{episodes} - Steps: {step_count}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
    
    # Save the trained model for this difficulty
    model.save(f"weights/{level}_dqn_model.h5")
    print(f"{level.capitalize()} model training complete and saved.")
    trained_models[level] = model  # store in dictionary if needed for later use
