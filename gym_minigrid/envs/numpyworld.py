from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np


from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np

COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

class NumpyMap(MiniGridEnv):
    """
    Environment created if given a numpy array and index mapping
    """

    def __init__(self):
        super().__init__(grid_size=19, max_steps=100)

    def _gen_grid(self, array, index_mapping):
        # Create the grid
        self.array = array

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                entity_name = index_mapping[array[i][j]]
                for worldobj in WorldObj.__subclasses__():
                    # if entity == worldobj.__name__:
                    #     print('entity')
                    entity = WorldObj.__subclasses__()[array[i][j]]()
                    self.put_obj(entity, i, j)


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info



class NumpyMapFourRooms(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, array, index_mapping, max_steps):
        self.array = array
        # super().__init__(grid_size=41, max_steps=1000)
        super().__init__(width=self.array.shape[1],
                         height= self.array.shape[0],
                         max_steps=max_steps)


    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # aux variable to see whether goal and agent where saved
        agent_defined = False
        goal_defined = False

        # Create the grid
        for i in range(1, self.array.shape[0]):
            for j in range(1, self.array.shape[1]):
                entity_name = self.index_mapping[self.array[i][j]]
                entity_index = int(self.array[i][j])

                if entity_index != 10 and entity_index !=8 and entity_index != 0:  #'agent' and 'unseen':
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            self.put_obj(entity_class(), j, i)

                # Place some WorldObj in Unseen, eg. Wall -- avoid locate agent in those points
                elif entity_index == 0:
                    self.put_obj(Wall(), j, i)

                # agent location
                elif entity_index == 10:
                    agent_x, agent_y = j,i # i=row=y
                    agent_defined = True
                    # print('Agent defined at file: [{},{}]'.format(agent_x,agent_y))

                # goal location
                elif entity_index == 8:
                    goal_x, goal_y = j,i
                    goal_defined = True
                    # print('Goal defined at file: [{},{}]'.format(goal_x,goal_y))

        # SET AGENT LOCATION
        if agent_defined:
            agent_position = self.place_agent(top=(agent_x,agent_y),deterministic_pos=True)
        else:
            agent_x, agent_y = 20,20
            agent_position = self.place_agent(top=(agent_x,agent_y),deterministic_pos=True)

        # SET GOAL LOCATION
        if goal_defined:
            goal_position = self.place_obj(obj=Goal(),
                                            top=(goal_x,goal_y),
                                            deterministic_pos=True)
        else:
            goal_x, goal_y = 24, 22
            goal_position = self.place_obj(obj=Goal(),
                                            top=(goal_x,goal_y),
                                            deterministic_pos=True)

        self.mission = 'Llegar al goal'#'Init position:{}\nReach the goal{}'.format(agent_position,goal_position)


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class NumpyMapMultiRoom(MiniGridEnv):
    """
    Environment with multiple rooms and random objects.
    This environment has no specific goals or rewards.
    """

    def __init__(self, array, index_mapping, max_steps):
        self.array = array
        print('dims: {}x{}'.format(array.shape[0],array.shape[1]))
        # super().__init__(grid_size=41, max_steps=1000)
        super().__init__(width=self.array.shape[1],
                         height= self.array.shape[0],
                         max_steps=max_steps)


    def _gen_grid(self, width, height):

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # aux variable to see whether goal and agent where saved
        agent_defined = False
        goal_defined = False

        # Create the grid
        for i in range(1, self.array.shape[0]):
            for j in range(1, self.array.shape[1]):
                entity_name = self.index_mapping[self.array[i][j]]
                entity_index = int(self.array[i][j])

                if entity_index != 10 and entity_index !=8 and entity_index!=4 and entity_index != 0 and entity_index!=9:  #'agent' and 'unseen':
                    for entity_class in WorldObj.__subclasses__():
                        # the class name needs to be lowercase (not sentence case)
                        if entity_name == entity_class.__name__.casefold():
                            # print('{} at :{},{}'.format(entity_class(),j,i))
                            self.put_obj(entity_class(), j, i)

                # Place some WorldObj in Unseen, eg. Wall -- avoid locate agent in those points
                elif entity_index == 0:
                    print('wall at :{},{}'.format(j,i))
                    self.put_obj(Wall(), j, i)

                elif entity_index == 9:
                    print('lava at :{},{}'.format(j,i))
                    self.put_obj(Lava(), j, i)

                # agent location
                elif entity_index == 10:
                    print('agent at :{},{}'.format(j,i))
                    agent_x, agent_y = j,i # i=row=y
                    agent_defined = True
                    # print('Agent defined at file: [{},{}]'.format(agent_x,agent_y))

                # goal location
                elif entity_index == 8:
                    print('goal at :{},{}'.format(j,i))
                    goal_x, goal_y = j,i
                    goal_defined = True
                    # print('Goal defined at file: [{},{}]'.format(goal_x,goal_y))

                elif entity_index == 4:
                    print('door at :{},{}'.format(j,i))
                    color = np.random.choice(list(COLOR_TO_IDX.keys()))
                    self.put_obj(Door(color=color), j, i)

        # SET AGENT LOCATION
        if agent_defined:
            agent_position = self.place_agent(top=(agent_x,agent_y),deterministic_pos=True)
        else:
            agent_x, agent_y = 20,20
            agent_position = self.place_agent(top=(agent_x,agent_y),deterministic_pos=True)

        # SET GOAL LOCATION
        if goal_defined:
            goal_position = self.place_obj(obj=Goal(),
                                            top=(goal_x,goal_y),
                                            deterministic_pos=True)
        else:
            goal_x, goal_y = 24, 22
            goal_position = self.place_obj(obj=Goal(),
                                            top=(goal_x,goal_y),
                                            deterministic_pos=True)

        self.mission = 'Llegar al goal'#'Init position:{}\nReach the goal{}'.format(agent_position,goal_position)


    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class NumpyMapFourRoomsPartialView(NumpyMapFourRooms):
	"""
	Assuming that `grid.npy` exists in the root folder
	with the approperiate shape for the grid (i.e. 40x40)
	"""
	def __init__(self, numpyFile='numpyworldfiles/map000.npy',max_steps=100):
		self.array = np.load(numpyFile)
		self.index_mapping = {
             0 : 'unseen'        ,
             1 : 'empty'         ,
             2 : 'wall'          ,
             3 : 'floor'         ,
             4 : 'door'          ,
             5 : 'key'           ,
             6 : 'ball'          ,
             7 : 'box'           ,
             8 : 'goal'          ,
             9 : 'lava'          ,
             10: 'agent'
        }
		super().__init__(self.array, self.index_mapping,max_steps)

class NumpyMapMultiRoomPartialView(NumpyMapMultiRoom):
	"""
	Assuming that `grid.npy` exists in the root folder
	with the approperiate shape for the grid (i.e. 40x40)
	"""
	def __init__(self, numpyFile='numpyworldfiles/MN3S8_test_key.npy',max_steps=100):
		self.array = np.load(numpyFile)
		self.index_mapping = {
             0 : 'unseen'        ,
             1 : 'empty'         ,
             2 : 'wall'          ,
             3 : 'floor'         ,
             4 : 'door'          ,
             5 : 'key'           ,
             6 : 'ball'          ,
             7 : 'box'           ,
             8 : 'goal'          ,
             9 : 'lava'          ,
             10: 'agent'
        }
		super().__init__(self.array, self.index_mapping,max_steps)



register(
    id='MiniGrid-NumpyMap-v0',
    entry_point='gym_minigrid.envs:NumpyMap'
)
register(
    id='MiniGrid-NumpyMapFourRooms-v0',
    entry_point='gym_minigrid.envs:NumpyMapFourRooms'
)
register(
	id='MiniGrid-NumpyMapFourRoomsPartialView-v0',
    entry_point='gym_minigrid.envs:NumpyMapFourRoomsPartialView'
)

register(
	id='MiniGrid-NumpyMapMultiRoom-v0',
    entry_point='gym_minigrid.envs:NumpyMapMultiRoom'
)
register(
	id='MiniGrid-NumpyMapMultiRoomPartialView-v0',
    entry_point='gym_minigrid.envs:NumpyMapMultiRoomPartialView'
)
