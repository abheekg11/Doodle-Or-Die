---
layout: default
title: Status
---

<iframe width="752" height="432" src="https://www.youtube.com/embed/wR9MilgXEm0" title="Why does Everyone Hate the BEST FISH?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Project Summary

Our project explores using reinforcement learning to train an agent for Doodle Jump. We are currently comparing various RL methods including DQN, A2C, and PPO to see how well they survive and how high they progress in Doodle Jump. While building a pipeline for training, we are also experimenting with tuning hyperparameters and plan to evaluate across different game difficulty levels.

## Approach

**Environment Setup, States, Actions, and Rewards**  

For our three reinforcement learning algorithms (A2C, PPO, DQN), the Doodle Jump environment provides a set of three actions which include moving left, moving right, or making no movement. At every step, the set of states will include a vector representing the position of the agent, a vector with the velocity of the agent, and a set of the positions of the platforms currently on the screen and their types. Also, springs and monsters are another part of the set of states.  

The environment uses an event driven reward in which the following specific events lead to rewards:  
- +3 if the score increases (progressing to higher platforms)
- +3 for hitting a spring 
- -2 if the agent dies or gets stuck (30 seconds of no progression)
- -4 for colliding with a monster
- 0 for any other case



## Evaluation

## Remaining Goals and Challenges

## Resources Used

## Video Summary

[![Watch our video on youtube](https://images.pexels.com/photos/128756/pexels-photo-128756.jpeg?_gl=1*198fe27*_ga*MTM5MjMxMjk4OS4xNzcxODkyNDkx*_ga_8JE65Q40S6*czE3NzE4OTI0OTEkbzEkZzEkdDE3NzE4OTI0OTYkajU1JGwwJGgw)](https://www.youtube.com/watch?v=wR9MilgXEm0)
