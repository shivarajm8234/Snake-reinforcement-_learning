import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import deque
from random import sample
import matplotlib.pyplot as plt
from IPython import display

# Constants
BLOCK_SIZE = 20
WIDTH = HEIGHT = 20 * BLOCK_SIZE
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 200)

# PyGame initialization
pygame.init()

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Snake Game Environment
class SnakeGameEnv:
    def __init__(self, width=WIDTH, height=HEIGHT, block_size=BLOCK_SIZE, speed=20, render=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed
        self.render_enabled = render
        
        # Initialize display if rendering is enabled
        if self.render_enabled:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
            
        # Game state variables
        self.reset()
    
    def reset(self):
        # Initial snake position and direction
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = random.choice([(0, -self.block_size), (0, self.block_size), 
                                   (-self.block_size, 0), (self.block_size, 0)])
        self.score = 0
        self.food = self._place_food()
        self.steps = 0
        self.steps_without_food = 0
        
        # Get initial state
        state = self._get_state()
        return state
    
    def _place_food(self):
        """Place food at random location not occupied by snake"""
        while True:
            x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
            food_pos = (x, y)
            if food_pos not in self.snake:
                return food_pos
    
    def _get_state(self):
        """Get the state representation for the agent"""
        head_x, head_y = self.snake[0]
        
        # Determine current direction (one-hot encoded)
        dir_l = self.direction == (-self.block_size, 0)
        dir_r = self.direction == (self.block_size, 0)
        dir_u = self.direction == (0, -self.block_size)
        dir_d = self.direction == (0, self.block_size)
        
        # Check for danger (collision would occur)
        point_l = (head_x - self.block_size, head_y)
        point_r = (head_x + self.block_size, head_y)
        point_u = (head_x, head_y - self.block_size)
        point_d = (head_x, head_y + self.block_size)
        
        # Calculate dangers based on current direction
        danger_straight = (dir_r and self._is_collision(point_r)) or \
                          (dir_l and self._is_collision(point_l)) or \
                          (dir_u and self._is_collision(point_u)) or \
                          (dir_d and self._is_collision(point_d))
        
        danger_right = (dir_r and self._is_collision(point_d)) or \
                       (dir_l and self._is_collision(point_u)) or \
                       (dir_u and self._is_collision(point_r)) or \
                       (dir_d and self._is_collision(point_l))
        
        danger_left = (dir_r and self._is_collision(point_u)) or \
                      (dir_l and self._is_collision(point_d)) or \
                      (dir_u and self._is_collision(point_l)) or \
                      (dir_d and self._is_collision(point_r))
        
        # Food location relative to snake head
        food_x, food_y = self.food
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y
        
        # Return state as array
        return np.array([
            # Dangers
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            dir_l,
            dir_r, 
            dir_u,
            dir_d,
            
            # Food location
            food_left,
            food_right,
            food_up,
            food_down
        ], dtype=np.float32)
    
    def _is_collision(self, point):
        """Check if point will cause collision"""
        x, y = point
        
        # Collision with boundaries
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        # Collision with self
        if point in self.snake[1:]:
            return True
        
        return False
    
    def _calculate_reward(self, old_state, new_state, game_over):
        """Calculate reward based on action results"""
        if game_over:
            return -20  # Penalty for death
        
        if self.snake[0] == self.food:
            return 15  # Reward for eating food
        
        # Calculate change in distance to food
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        old_distance = math.sqrt((self.prev_head[0] - food_x)**2 + (self.prev_head[1] - food_y)**2)
        new_distance = math.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
        
        # Reward for moving closer to food, penalty for moving away
        if new_distance < old_distance:
            return 1
        else:
            return -1.5
    
    def step(self, action):
        """
        Execute one time step within the environment
        action: 0 = straight, 1 = right turn, 2 = left turn
        """
        # Process events (allow window to be closed)
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # Store current head position for reward calculation
        self.prev_head = self.snake[0]
        
        # Change direction based on action
        # 0: keep direction, 1: turn right, 2: turn left
        if action == 1:  # right turn
            if self.direction == (self.block_size, 0):
                self.direction = (0, self.block_size)
            elif self.direction == (0, self.block_size):
                self.direction = (-self.block_size, 0)
            elif self.direction == (-self.block_size, 0):
                self.direction = (0, -self.block_size)
            elif self.direction == (0, -self.block_size):
                self.direction = (self.block_size, 0)
        elif action == 2:  # left turn
            if self.direction == (self.block_size, 0):
                self.direction = (0, -self.block_size)
            elif self.direction == (0, -self.block_size):
                self.direction = (-self.block_size, 0)
            elif self.direction == (-self.block_size, 0):
                self.direction = (0, self.block_size)
            elif self.direction == (0, self.block_size):
                self.direction = (self.block_size, 0)
        
        # Get current state before moving
        old_state = self._get_state()
        
        # Move snake
        x, y = self.snake[0]
        dx, dy = self.direction
        new_head = (x + dx, y + dy)
        self.snake.insert(0, new_head)
        
        # Check for collision
        game_over = self._is_collision(self.snake[0])
        
        # Check for food
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._place_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            self.steps_without_food += 1
        
        # Get new state after moving
        new_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(old_state, new_state, game_over)
        
        # Increment step counter
        self.steps += 1
        
        # Render if enabled
        if self.render_enabled:
            self._render()
        
        # Early stopping for efficiency
        if self.steps_without_food > 100 * len(self.snake):
            game_over = True
            reward = -10  # Penalty for not finding food efficiently
        
        return new_state, reward, game_over, {"score": self.score}
    
    def _render(self):
        """Render the game state"""
        if not self.render_enabled or self.display is None:
            return
            
        self.display.fill(BLACK)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = BLUE if i == 0 else GREEN  # Head is different color
            pygame.draw.rect(self.display, color, pygame.Rect(x, y, self.block_size, self.block_size))
        
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        # Draw score
        font = pygame.font.SysFont('arial', 16)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(score_text, [0, 0])
        
        # Update display
        pygame.display.update()
        self.clock.tick(self.speed)
    
    def set_render(self, render):
        """Enable or disable rendering"""
        if render and not self.render_enabled:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
            self.render_enabled = True
        elif not render and self.render_enabled:
            self.render_enabled = False

# Replay Memory
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else self.buffer
    
    def __len__(self):
        return len(self.buffer)

# Agent
class DQNAgent:
    def __init__(self, state_size=11, action_size=3, hidden_size=256, lr=0.001, 
                 gamma=0.9, epsilon=0.8, epsilon_min=0.05, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr
        
        # Networks
        self.policy_net = DQN(state_size, hidden_size, action_size)
        self.target_net = DQN(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Replay memory
        self.memory = ReplayBuffer()
        self.batch_size = 64
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy strategy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self):
        """Train the agent with experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values using target net
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        next_q_values[dones] = 0.0
        expected_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
    
    def save(self, filename="snake_dqn_model.pth"):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load(self, filename="snake_dqn_model.pth"):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

# Training function
def train_agent(episodes=1000, render_interval=100, target_update=10):
    env = SnakeGameEnv(render=False, speed=100)
    agent = DQNAgent()
    
    scores = []
    avg_scores = []
    losses = []
    steps_survived = []
    best_score = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        
        # For visualization, render every 'render_interval' episodes
        should_render = episode % render_interval == 0
        env.set_render(should_render)
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Store in replay memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
        
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Track metrics
        scores.append(info["score"])
        steps_survived.append(env.steps)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Calculate moving average score
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        # Save best model
        if info["score"] > best_score:
            best_score = info["score"]
            agent.save("best_snake_model.pth")
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {info['score']}, Average Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}, Steps: {env.steps}")
        
        # Visualization every 100 episodes
        if episode % 100 == 0 and episode > 0:
            try:
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(scores)
                plt.plot(avg_scores)
                plt.title('Score')
                plt.xlabel('Episode')
                plt.ylabel('Score')
                plt.legend(['Score', 'Avg Score'])
                
                plt.subplot(1, 3, 2)
                plt.plot(losses)
                plt.title('Loss')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
                
                plt.subplot(1, 3, 3)
                plt.plot(steps_survived)
                plt.title('Steps Survived')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                
                plt.tight_layout()
                try:
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                except:
                    plt.savefig(f'training_progress_{episode}.png')
                plt.close()
            except Exception as e:
                print(f"Visualization error: {e}")
    
    # Final plot
    try:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(scores)
        plt.plot(avg_scores)
        plt.title('Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend(['Score', 'Avg Score'])
        
        plt.subplot(1, 3, 2)
        plt.plot(losses)
        plt.title('Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(steps_survived)
        plt.title('Steps Survived')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
    except Exception as e:
        print(f"Final visualization error: {e}")
    
    return agent, scores, losses, steps_survived

# Test trained agent
def test_agent(agent, episodes=10, speed=10):
    env = SnakeGameEnv(render=True, speed=speed)
    scores = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, _, done, info = env.step(action)
        scores.append(info["score"])
        print(f"Test Score: {info['score']}")
    
    print(f"Average Test Score: {np.mean(scores)}")
    pygame.quit()

# Main execution
if __name__ == "__main__":
    # Train the agent
    print("Starting training...")
    agent, scores, losses, steps = train_agent(episodes=2000)
    
    # Save final model
    agent.save("final_snake_model.pth")
    
    # Test the agent
    print("Testing trained agent...")
    test_agent(agent)