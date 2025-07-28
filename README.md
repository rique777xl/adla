Adla Game Economic Control
This project aims to develop an Artificial Intelligence (AI) capable of simulating and controlling a game's economy, with a focus on adaptability and continuous learning. The AI, named Adla, acts as an economic "brain," analyzing the current state of the game and making decisions (such as adjustments to the withdrawal rate) to optimize key economic metrics.

How It Works
Adla operates in a continuous cycle of observation, decision, simulation, and learning:

Current State Reading: Adla reads various economic parameters of the game (transaction volume, NFT prices, daily active users, liquidity, token price, etc.) from text files (.txt).

Analysis and Decision: Based on this data, Adla processes the current state of the economy. It uses reinforcement learning models to decide the best action to take, such as adjusting the withdrawal rate.

Simulation with Gemini: Adla's decision (the new withdrawal rate) is then sent to an external simulator. Currently, we use the Gemini model as a simulation "brain." Gemini receives the current economic state and the withdrawal rate defined by Adla, and then projects how the economy would behave in the next cycle, returning the new values for all economic parameters.

Update and Learning: The new values simulated by Gemini are written back to the .txt files, updating the economy's state for the next cycle. Adla then records this experience (previous state, action taken, reward generated, new state) in its replay memory. By accumulating experiences, Adla trains its internal models to improve its future decisions.

Pre-Loaded Models (Under Training)
This project uses deep learning models (LSTM neural networks and reinforcement learning models) that are saved in the .h5 format. When starting the script, Adla attempts to load these pre-existing models.

Important: Currently, these .h5 models are in a phase of continuous training. They are still learning and adapting to the dynamics of the simulated economy. Therefore, Adla's initial decisions may not be optimal, and its performance is expected to improve as more simulation data is accumulated and training progresses.

Current Status: Test Model and Training via TXT
At the moment, the project is configured as a test and training model. Communication between Adla and the simulator (Gemini) is done through text files (.txt). This allows for easy and transparent data control for development and debugging purposes.

The persistence of data in .txt files is an intentional step to facilitate the training process and the visualization of information flow. In a future production phase, communication would be migrated to APIs for greater efficiency and scalability.

Dependencies
To run this project, you will need Python 3.10 and the following libraries:

pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install requests 


Next Steps
The goal is that, after extensive training with simulated data and, subsequently, with real game data such as Axie Infinity, Adla will become a robust and autonomous tool for managing game economies. The open-source nature of the project will allow community collaboration in defining and refining economic goals, making Adla adaptable to various game scenarios.
