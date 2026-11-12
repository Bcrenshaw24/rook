# rook

A player scoring engine that designates NBA players a given score over the course of a game. The scoring mechanism is inspired by Chess.com and their ELO rating. 

## Archictecture 

### Tracking 

We analyze a given game using OpenCV to allow us to track players and their movements. 

### Scoring 

We train an RL model on historical data to develop a "rule engine". Instead of measuring if a game is won, we attempt to opimize a Reward-Optimized On-court Knowledge (ROOK) function.

An initial idea of how this is scored is: 

$$
\text{ROOK}(s, a) = w_1 x_1 + w_2 x_2 + \dots + b
$$


## Phases 

### Phase 1 - Tracking (Ongoing)

- Gather a few clips for analysis 
- Build a tracking module with OpenCV/YOLO 
- Test module on different clips 
- Determine what can be done with analyzed clips 
- Analyze shot and pass trajectory 

### Phase 2 - State mapping 

- Map tracking output to court coordinates 
- Attach ball and trajectory to mapping 
- Attach clock and player stats 

### Phase 3 - Modeling 

- Train a model to predict how a play contributes to a win 
- Break down each play into individual player actions and contributions 
- Score each decision against alternatives using MCTS over the model

### Phase 4 - Player ELO 

- Determine player ROOK rating based on past performance 
- Determine how it updates over games 

### Phase 5 - Testing and Analysis 

- Implement and observe different scoring and modeling methods 
- Run past seasons, look for trends, and project winners 
- Build a comprehensive database of player scores based on current data 