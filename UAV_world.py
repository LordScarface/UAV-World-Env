import gym
from gym import spaces
import pygame
import numpy as np
import sys

# half the communication range, 5 actors
# increase the reward for updating the operator

class UAV_world(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode='rgb_array', size=20, num_actors=5, agent_range=3, operator_range=3, max_steps=1000, message=None):
        self.size = size  # The size of the square grid
        self.window_size = 800  # The size of the PyGame window

        self.num_actors = num_actors
        self.agent_range = agent_range
        self.operator_range = operator_range
        self.max_steps = max_steps
        self.message = message

        self.action_space = spaces.MultiDiscrete([4 for _ in range(self.num_actors)])
        #self.action_space = spaces.Box(0, 3, shape=(num_actors,), dtype=int)

        # maybe add distance to goal, sth about reward https://stackoverflow.com/questions/73922332/dict-observation-space-for-stable-baselines3-not-working
        self.observation_space = spaces.Dict()

        self.observation_space["agent_map"] = spaces.Box(0, 2, shape=(num_actors,size,size), dtype=int)

        self.observation_space["agent_pos"] = spaces.Box(0, size-1, shape=(num_actors,2), dtype=int) #spaces.Tuple((spaces.Discrete(size), spaces.Discrete(size)))


        
        
        self.gt_map = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.]]).T
        
        """
        # thick walls:
        self.gt_map = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 1., 1.],
                                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,0., 0., 0., 1.],
                                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,0., 0., 0., 0.],
                                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],
                                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.,0., 0., 0., 0.],
                                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.]]).T
        """
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        _obs = {}

        #for agent in range(self.num_actors):
        #_obs["agent_{}".format(agent)] = np.array(self._agent_maps[agent], dtype=int) * 0.5
        _obs["agent_map"] = np.array(self._agent_maps, dtype=int)
        #_obs["reward_map"] = np.array(self.history, dtype=float)
        _obs["agent_pos"] = np.array(self._agent_location, dtype=int)
        

        return _obs

    def _get_info(self):
        return {
            "distance": np.array([np.linalg.norm(
                agent - self._target_location, ord=1
            ) for agent in self._agent_location]),
            "positions": self._agent_location
            #"terminal_observation": self._get_obs(),
            #"TimeLimit.truncated": False
        }

    def reset(self):
        """
            Reset the environment and initialize all variables
            num_actors: number of UAVs
            agent_range: communication range of agents for map updates
            operator_range: communication range of the operator for map updates
        """
        self._agent_colors = np.array([[np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255)] for _ in range(self.num_actors)])
        
        # We will sample the target's location randomly until it does not coincide with the agent's location
        #self._target_location = np.array([18,3])
        target_blocked = True
        while target_blocked:
            target_x = np.random.randint(1,19)
            target_y = np.random.randint(1,16)

            if self.gt_map[target_x, target_y] == 0:
                target_blocked = False
                
        self._target_location = np.array([target_x,target_y])
        self._target_location = np.array([18,3])
        self._operator_position = np.array([2,18])
        
        # operator and agent knowledge about position of the swarm
        self._agent_location_agent_current = np.zeros((self.num_actors, self.num_actors, 2))
        self._agent_location_agent_current_obersavton_time = np.zeros((self.num_actors, self.num_actors))
        self._agent_location_operator_current = np.zeros((self.num_actors, 2))
        
        # discovered posions: discovering a new square gives a reward, going to the same squares repeatedly is bad
        self.history = np.ones((self.num_actors,self.size,self.size)) * 0.5

        # operator start with coarse map, just goal position is visible
        self._operator_map = np.zeros((self.size, self.size))
        self._operator_map[self._target_location[0], self._target_location[1]] = 2

        # agents get the initial map from operator
        self._agent_maps = np.zeros((self.num_actors, self.size, self.size))
        for i in range(self.num_actors):
            self._agent_maps[i,self._target_location[0], self._target_location[1]] = 2

        # mark goal on gt map
        self.gt_map[self._target_location[0], self._target_location[1]] = 2

        # Agents spawn at operator
        self._agent_location = np.array([self._operator_position for _ in range(self.num_actors)])

        # each actor gets an ID
        self._actor_UID = np.array([i for i in range(self.num_actors)])

        # count number of steps taken
        self._steps_taken = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation#, info

    def set_render_mode(self, mode):
        self.render_mode = mode

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we fly in
        #direction = [self._action_to_direction[np.round(a)] for a in action]
        direction = [self._action_to_direction[a] for a in action]

        # increment steps taken
        self._steps_taken += 1

        # reset the reward
        reward = 0

        # remember if operator received update in this step and how large the update was
        operator_update = False
        operator_update_score = 0 

        # np.clip to make sure we don't leave the grid
        tmp_location = self._agent_location
        for ix,agent in enumerate(tmp_location):

            # check for collision with boundry or walls
            new_pos = np.array(agent + direction[ix])
            if new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size or self.gt_map[new_pos[0], new_pos[1]] == 1:
                reward -= 1

            agent = np.clip(
                agent + direction[ix], 0, self.size - 1
            )

            # handle collision
            if self.gt_map[agent[0], agent[1]] == 1:
                agent = tmp_location[ix]

            self._agent_location[ix] = agent 
            
            # reward for discovering new square or punish stepping on previously visited square
            reward += self.history[ix,self._agent_location[ix][0], self._agent_location[ix][1]]
            self.history[ix,self._agent_location[ix][0], self._agent_location[ix][1]] -= 0.5 # decrese the reward for next time
            if self.history[ix,self._agent_location[ix][0], self._agent_location[ix][1]] <= -1:
                self.history[ix,self._agent_location[ix][0], self._agent_location[ix][1]] = -1 # but cap at -1

            # save agent map before update
            agent_old_map = self._agent_maps[ix]

            # handle map updates
            for x in range(self._agent_location[ix][0] - self.agent_range, self._agent_location[ix][0] + self.agent_range):
                for y in range(self._agent_location[ix][1] - self.agent_range, self._agent_location[ix][1] + self.agent_range):
                    if x > 0 and x < self.size - 1 and y > 0 and y < self.size - 1:
                        # reveal a piece of the map
                        self._agent_maps[ix,x,y] = self.gt_map[x,y] 

            # give a reward for each new map part that in discovered
            reward += np.sum(np.abs(self._agent_maps[ix] - agent_old_map))

        # handle communication
        for ix,agent in enumerate(self._agent_location):
            # each agent observes themselves every time
            self._agent_location_agent_current[ix,ix] = agent      
            self._agent_location_agent_current_obersavton_time[ix,ix] = self._steps_taken
            
            # with other agents
            for iy,other_agent in enumerate(self._agent_location):
                # if we are in communication range
                if np.sqrt((agent[0] - other_agent[0])**2 + (agent[1] - other_agent[1])**2) < self.agent_range and ix != iy:
                    # update the internal map
                    self._agent_maps[ix] = self._agent_maps[ix] + self._agent_maps[iy]
                    self._agent_maps[ix, self._agent_maps[ix] == 2] = 1
                    self._agent_maps[ix, self._agent_maps[ix] == 4] = 2
                    
                    # update each other about positions on the map
                    most_up_to_date = np.zeros((self.num_actors,2))
                    for iz in range(self.num_actors):
                        if self._agent_location_agent_current_obersavton_time[ix,iz] >= self._agent_location_agent_current_obersavton_time[iy,iz]:
                            most_up_to_date[iz] = self._agent_location_agent_current[ix,iz]
                        else:
                            most_up_to_date[iz] = self._agent_location_agent_current[iy,iz]
                            
                    most_up_to_date[ix] = self._agent_location[ix]
                    most_up_to_date[iy] = self._agent_location[iy]
                    
                    #print(most_up_to_date)   
                    self._agent_location_agent_current[ix] = most_up_to_date
                    self._agent_location_agent_current[iy] = most_up_to_date
                    
                    self._agent_location_agent_current_obersavton_time[ix,iy] = self._steps_taken
                    self._agent_location_agent_current_obersavton_time[iy,ix] = self._steps_taken
                    
                    # share the history of discovered tiles
                    # elementwise minimum of the two histories
                    shared_history = np.minimum(self.history[ix], self.history[iy])
                    self.history[ix] = shared_history
                    self.history[iy] = shared_history

            # with operator
            if np.sqrt((agent[0] - self._operator_position[0])**2 + (agent[1] - self._operator_position[1])**2) < self.operator_range:
                # update operators map
                operator_update_score += np.sum(np.abs(self._operator_map - self._agent_maps[ix]))
                self._operator_map = self._operator_map + self._agent_maps[ix]
                self._operator_map[self._operator_map == 2] = 1
                self._operator_map[self._operator_map == 4] = 2
                operator_update = True
                reward += operator_update_score * 10

        # operator update agents
        # handle communication
        for ix,agent in enumerate(self._agent_location):
            if np.sqrt((agent[0] - self._operator_position[0])**2 + (agent[1] - self._operator_position[1])**2) < self.operator_range:
                # update operators map
                self._agent_maps[ix] = self._operator_map
                
                # update operator about most recent positon update of other actors
                self._agent_location_operator_current = self._agent_location_agent_current[ix]

        # An episode is done iff the agent has reached the target
        terminated = any([np.array_equal(ag, self._target_location) for ag in self._agent_location])
        # reward = 1 if terminated else 0  # Binary sparse rewards

        if terminated:
            # large reward for reaching the goal, but scale with number of steps taken 
            # TODO: check what works better
            # reward += (100 * (self.max_steps - self._steps_taken) )# (self._steps_taken / self.max_steps) )
            reward += (100 * (self._steps_taken / self.max_steps) )
            self._steps_taken = 0

        # define the reward
        #reward = int(terminated) #- 0.001 * self._num_goal_steps #* 10 + operator_update_score
        
        if terminated:
            observation = self.reset()
            info = self._get_info()
        else:
            observation = self._get_obs()
            info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self, mode=None, **kwargs):
        if mode is not None:
            if mode == "rgb_array":
                    return self._render_frame()
            elif mode == "human":
                self._render_frame()

        elif self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size+(250 * (np.ceil((self.num_actors - 2) / 3) + 1)), self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size+(250 * (np.ceil((self.num_actors - 2) / 3) + 1)), self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw the operator map
        for i in range(self.size):
            for j in range(self.size):
                if self._operator_map[i,j] == 0:
                    pygame.draw.rect(
                        canvas,
                        (200,200,200),
                        pygame.Rect(
                            10 * np.array([82+i,55+j]),
                            (10, 10),
                        ),
                    )
                    for agent in range(self.num_actors):
                        if self._agent_location_operator_current[agent][0] == i and self._agent_location_operator_current[agent][1] == j:
                            pygame.draw.rect(
                                canvas,
                                (200,200,200), # self._agent_colors[agent],
                                pygame.Rect(
                                    10 * np.array([82+i,55+j]),
                                    (10, 10),
                                ),
                            )
                elif self._operator_map[i,j] == 2:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            10 * np.array([82+i,55+j]),
                            (10, 10),
                        ),
                    )
                else:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            10 * np.array([82+i,55+j]),
                            (10, 10),
                        ),
                    )

        if self.num_actors > 1:
            for i in range(self.size):
                for j in range(self.size):
                    if self._agent_location[1,0] == i and self._agent_location[1,1] == j:
                        pygame.draw.rect(
                            canvas,
                            self._agent_colors[1],
                            pygame.Rect(
                                10 * np.array([82+i,30+j]),
                                (10, 10),
                            ),
                        )
                    elif self._agent_maps[1,i,j] == 0:
                        if self.history[1,i,j] == 0.5:
                            reward_color = (152,228,125)
                        elif self.history[1,i,j] == 0:
                            reward_color = (221,228,125)
                        elif self.history[1,i,j] == -0.5:
                            reward_color = (228,193,125)
                        elif self.history[1,i,j] == -1:
                            reward_color = (228,125,125)
                        pygame.draw.rect(
                            canvas,
                            reward_color,
                            pygame.Rect(
                                10 * np.array([82+i,30+j]),
                                (10, 10),
                            ),
                        )
                    elif self._agent_maps[1,i,j] == 2:
                        pygame.draw.rect(
                            canvas,
                            (255, 0, 0),
                            pygame.Rect(
                                10 * np.array([82+i,30+j]),
                                (10, 10),
                            ),
                        )
                    else:
                        pygame.draw.rect(
                            canvas,
                            (0, 0, 0),
                            pygame.Rect(
                                10 * np.array([82+i,30+j]),
                                (10, 10),
                            ),
                        )

        # Draw the agent map 0
        for i in range(self.size):
            for j in range(self.size):
                if self._agent_location[0,0] == i and self._agent_location[0,1] == j:
                    pygame.draw.rect(
                        canvas,
                        self._agent_colors[0],
                        pygame.Rect(
                            10 * np.array([82+i,5+j]),
                            (10, 10),
                        ),
                    )
                elif self._agent_maps[0,i,j] == 0:
                    if self.history[0,i,j] == 0.5:
                        reward_color = (152,228,125)
                    elif self.history[0,i,j] == 0:
                        reward_color = (221,228,125)
                    elif self.history[0,i,j] == -0.5:
                        reward_color = (228,193,125)
                    elif self.history[0,i,j] == -1:
                        reward_color = (228,125,125)
                    pygame.draw.rect(
                        canvas,
                        reward_color,
                        pygame.Rect(
                            10 * np.array([82+i,5+j]),
                            (10, 10),
                        ),
                    )
                elif self._agent_maps[0,i,j] == 2:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            10 * np.array([82+i,5+j]),
                            (10, 10),
                        ),
                    )
                else:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            10 * np.array([82+i,5+j]),
                            (10, 10),
                        ),
                    )

        if self.num_actors > 2:
            for ag in range(2, self.num_actors):
                # Draw the agent map 
                for i in range(self.size):
                    for j in range(self.size):
                        if self._agent_location[ag,0] == i and self._agent_location[ag,1] == j:
                                pygame.draw.rect(
                                canvas,
                                self._agent_colors[ag],
                                pygame.Rect(
                                    10 * np.array([82+i+(np.floor((ag-2) / 3)+1)*25,5+j+(ag-2)%3*25]),
                                    (10, 10),
                                ),
                            )
                        elif self._agent_maps[ag,i,j] == 0:
                            if self.history[ag,i,j] == 0.5:
                                reward_color = (152,228,125)
                            elif self.history[ag,i,j] == 0:
                                reward_color = (221,228,125)
                            elif self.history[ag,i,j] == -0.5:
                                reward_color = (228,193,125)
                            elif self.history[ag,i,j] == -1:
                                reward_color = (228,125,125)
                            pygame.draw.rect(
                                canvas,
                                reward_color,
                                pygame.Rect(
                                    10 * np.array([82+i+(np.floor((ag-2) / 3)+1)*25,5+j+(ag-2)%3*25]),
                                    (10, 10),
                                ),
                            )
                        elif self._agent_maps[ag,i,j] == 2:
                            pygame.draw.rect(
                                canvas,
                                (255, 0, 0),
                                pygame.Rect(
                                    10 * np.array([82+i+(np.floor((ag-2) / 3)+1)*25,5+j+(ag-2)%3*25]),
                                    (10, 10),
                                ),
                            )
                        else:
                            pygame.draw.rect(
                                canvas,
                                (0, 0, 0),
                                pygame.Rect(
                                    10 * np.array([82+i+(np.floor((ag-2) / 3)+1)*25,5+j+(ag-2)%3*25]),
                                    (10, 10),
                                ),
                            )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the operator
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._operator_position,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the walls
        for i in range(self.size):
            for j in range(self.size):
                if self.gt_map[i,j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([i,j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent
        for ix,agent in enumerate(self._agent_location):
            pygame.draw.circle(
                canvas,
                self._agent_colors[ix],
                (agent + 0.5) * pix_square_size,
                pix_square_size / 6,
            )
            pygame.draw.line(
                canvas,
                self._agent_colors[ix],
                ((agent + 0.2) * self.window_size / self.size),
                ((agent + 0.8) * self.window_size / self.size),
                width=4,
            )

            pygame.draw.line(
                canvas,
                self._agent_colors[ix],
                ((np.array([agent[0] - 0.2 + 1, agent[1] + 0.2])) * self.window_size / self.size),
                ((np.array([agent[0] - 0.8 + 1, agent[1] + 0.8])) * self.window_size / self.size),
                width=4,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            #pygame.font.init()
            self.window.blit(canvas, canvas.get_rect())
            if self.message is not None:
                GAME_FONT = pygame.font.SysFont("Arial", 16)
                surface = GAME_FONT.render(self.message, True, (0, 0, 0))
                self.window.blit(surface, (825,15))


            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
