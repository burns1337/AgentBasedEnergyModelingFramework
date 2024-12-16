import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Define the environment
class CoolingControlEnv:
    def __init__(self, building_data, cooling_setpoint, temperature_range):
        self.building_data = building_data
        self.cooling_setpoint = cooling_setpoint
        self.temperature_range = temperature_range
        self.state_dim = 2  # temperature and cooling setpoint
        self.action_dim = 1  # cooling rate
        self.max_cooling_rate = 1000  # W/m^2
        self.min_cooling_rate = 0  # W/m^2
        self.episode_length = 360  # seconds

    def reset(self):
        self.state = np.array(
            [self.building_data['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'][0], self.cooling_setpoint])
        return self.state

    def step(self, action):
        # Calculate the new temperature based on the cooling rate
        temperature = self.state[0] - action
        print(f"Temperature: {temperature}")
        temperature = np.clip(temperature, self.temperature_range[0], self.temperature_range[1])
        print(f"Temperature: {temperature}")

        # Calculate the reward
        reward = -abs(temperature - self.cooling_setpoint)
        done = False
        if temperature < self.temperature_range[0]:
            done = True
        elif temperature > self.temperature_range[1]:
            done = True

        # Update the state
        temperature = float(temperature)
        # print(f'Temperature: {temperature}, Cooling Setpoint: {self.cooling_setpoint}')
        self.state = np.array([temperature, self.cooling_setpoint])
        print(f"stage, reward, done: {self.state}, {reward}, {done}")
        return self.state, reward, done, {}


# Load the building data
building_data = pd.read_csv('../src/predictions/final_office_IdealLoad_summer_tiny_hourly_own_schedule_24_final_2.csv')
# take only columns of interest
building_data = building_data[['Date/Time', 'THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)',
                               'THERMAL ZONE 1:Zone Thermostat Cooling Setpoint Temperature [C](Hourly)']]
# take only the first 3 days of data
building_data = building_data.iloc[:12]
print(building_data)

# Create the environment
env = CoolingControlEnv(building_data, 21, (20, 22))

num_episodes = 3


# Define the RL agent
class RLAgent:
    def __init__(self, env, learning_rate, discount_factor, exploration_rate):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.env.state_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.env.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        print(model.summary())
        return model

    def act(self, state):
        state = np.reshape(state, (1, -1))  # Reshape the state to (1, state_dim)
        if np.random.rand() < self.exploration_rate:
            return np.random.uniform(self.env.min_cooling_rate, self.env.max_cooling_rate)
        else:
            return self.model.predict(state)[0]

    def learn(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, -1))  # Reshape the state to (1, state_dim)
        next_state = np.reshape(next_state, (1, -1))  # Reshape the next_state to (1, state_dim)
        target = reward + self.discount_factor * self.model.predict(next_state)[0]
        self.model.fit(state, target, epochs=1, verbose=0)


# Train the agent
agent = RLAgent(env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)
early_stopping = EarlyStopping(monitor='loss', patience=3)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        rewards += reward
    print(f'Episode {episode + 1}, Reward: {rewards}')


# Deploy the agent
def deploy_agent(env, agent):
    state = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
    return state


state = deploy_agent(env, agent)
print(f'Final temperature: {state[0]}')
