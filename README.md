# SC2-AI
Attempt to replace the Q learning algorithm used in this [tutorial](https://github.com/skjb/pysc2-tutorial/blob/master/Refining%20the%20Sparse%20Reward%20Agent/refined_agent.py) by DQN.

-	The inputs of the neural network are the different variables representing the state of the game.
-	The outputs are the differents Q values of each smart actions given a state.

**The project is currently a failure :** I still haven't found the hyperparameters to make this network functional. It is possible that there is a problem elsewhere in the code (changes had to be made on the PYSC2 side to be able to change the alhorithm used by the agent)

The main problem is that, since AI never wins and its goal is to maximize rewards, it cannot begin its learning process.
I tried to fix the problem by replacing the initial rewards 1 (win) / 0 (lose) by : final score * 5 (win) and final score / 5 (lose) but I don't feel like the AI plays better.

## Dependencies
-	pysc2
-	keras
-	tensorflow
-	hdf5 (to load and save a model)

## Areas for improvement
-	Obviously, increase the action space. With 4 possible "smart" actions, you'll never get far
-	Use the camera layers with a convolutional network to represent the game state
