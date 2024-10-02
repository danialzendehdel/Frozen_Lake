import sys
import gymnasium as gym
from gymnasium import spaces
import random
import pygame

class CustomSFLEnv(gym.Env):
    def __init__(self):
        super(CustomSFLEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: left, 1: down, 2: right, 3: up
        self.observation_space = spaces.Discrete(16)  # Grid 4 * 4

        # Define holes and goal cells
        self.holes = [5, 7, 11, 12]
        self.goal = 15
        self.grid_size = 4

        # Initial state
        self.state = 0
        self.step_count = 0

        # For rendering
        self.cell_size = 100  # Size of each cell in pixels
        self.screen_size = self.grid_size * self.cell_size  # Total screen size
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Parameters:
            seed: Optional[int] = None
                Sets the seed for the environment's random number generator(s) to ensure reproducibility.

            options: Optional[dict] = None
                Allows you to pass additional parameters or configuration options to customize the reset behavior.

        Returns:
            observation: The initial observation after resetting the environment.
            info: A dictionary containing additional information.
        """
        if seed is not None:
            random.seed(seed)
        # Set the initial state
        if options is not None and 'state' in options:
            self.state = options['state']
        else:
            self.state = 0  # Default initial state
        self.step_count = 0  # Reset step count
        return self.state, {}

    def step(self, action):
        """
        Processes an action and returns the new state, reward, and status.

        Parameters:
            action: The action taken by the agent.

        Returns:
            observation: The new state after taking the action.
            reward: The reward received for taking the action.
            done: A boolean indicating if the episode has ended.
            truncated: A boolean indicating if the episode was truncated.
            info: A dictionary containing additional information.
        """
        assert self.action_space.contains(action), f"{action} is not a valid action"

        self.step_count += 1
        action_probability = {
            0: [0, 1, 3],   # Left: intended left, or slips to down or up
            1: [1, 0, 2],   # Down: intended down, or slips to left or right
            2: [2, 1, 3],   # Right: intended right, or slips to down or up
            3: [3, 0, 2]    # Up: intended up, or slips to left or right
        }

        probs = [1/3, 1/3, 1/3]

        possible_actions = action_probability[action]
        actual_action = random.choices(possible_actions, weights=probs, k=1)[0]

        # Update position
        row = self.state // self.grid_size
        col = self.state % self.grid_size

        if actual_action == 0:  # Left
            col = max(0, col - 1)
        elif actual_action == 1:  # Down
            row = min(self.grid_size - 1, row + 1)
        elif actual_action == 2:  # Right
            col = min(self.grid_size - 1, col + 1)
        elif actual_action == 3:  # Up
            row = max(0, row - 1)

        new_state = row * self.grid_size + col
        self.state = new_state

        if new_state in self.holes:
            reward = 0
            done = True
        elif new_state == self.goal:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        truncated = False  # No truncation logic implemented
        info = {
            'step_count': self.step_count,
            'actual_action': actual_action  # Optional
        }

        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        if self.screen is None:
            # Initialize Pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Custom FrozenLake')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 0, 255)  # Holes
        GREEN = (0, 255, 0)  # Goal
        RED = (255, 0, 0)  # Agent
        GRAY = (200, 200, 200)  # Obstacles (if any)

        # Clear the screen
        self.screen.fill(WHITE)

        # Draw grid and elements
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.cell_size
                y = row * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Cell border

                state = row * self.grid_size + col

                # Draw holes
                if state in self.holes:
                    pygame.draw.rect(self.screen, BLUE, rect)
                # Draw goal
                elif state == self.goal:
                    pygame.draw.rect(self.screen, GREEN, rect)
                # Draw obstacles (if implemented)
                elif hasattr(self, 'obstacles') and state in self.obstacles:
                    pygame.draw.rect(self.screen, GRAY, rect)

        # Draw the agent
        agent_row = self.state // self.grid_size
        agent_col = self.state % self.grid_size
        agent_x = agent_col * self.cell_size + self.cell_size // 2
        agent_y = agent_row * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.screen, RED, (agent_x, agent_y), self.cell_size // 4)

        # Display additional information (e.g., step count)
        step_text = self.font.render(f"Steps: {self.step_count}", True, BLACK)
        self.screen.blit(step_text, (10, 10))

        # Update the display
        pygame.display.flip()
        self.clock.tick(10)  # Control the frame rate

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Testing the environment
if __name__ == "__main__":
    env = CustomSFLEnv()
    observation, info = env.reset()
    done = False

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)
        if done:
            env.render()
            if reward == 1:
                print("Reached the goal!")
            else:
                print("Fell into a hole!")
            print(f"Episode finished in {info['step_count']} steps.")
            pygame.time.wait(2000)  # Wait before closing
            break

    env.close()
