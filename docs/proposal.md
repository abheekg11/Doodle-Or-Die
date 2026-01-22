---
layout: default
title: Proposal
---

## Summary of the Project 
In our project, Doodle or Die, we will train a reinforcement learning agent to play the game Doodle Jump. Doodle Jump is a 2-D game where the player must make real time decisions to maximize height without falling off of various static and dynamic platforms and obstacles. The RL application of this project is in learning how to strategically jump and avoid behaviours that lead to falling or obstacle collision. At each step, the agent takes the current game state as input, including the position and shape of platforms, obstacles, and the player. Accordingly, it will continually choose a direction and magnitude along the horizontal axis. The overarching goal is to train an agent to go as high as possible both consistently and efficiently. 

## Project Goals 
**Minimum Goal:** 
- Implement an agent to successfully interact with Doodle Jump through simulated keyboard inputs, performing valid movements and surviving longer than a chosen baseline.
**Realistic Goal:** 
- Implement an agent that learns effective jumping, including targeting reachable platforms and avoiding gaps, performing consistently better than a simple heuristic based agent.
**Moonshot Goal:** 
- Implement an agent that demonstrates smooth control and consistently high scoring gameplay, including maximizing collection of power ups, avoiding monsters and breaking platforms, and adapting to new obstacles.

## AI/ML Algorithms 
We plan to use model-free, on-policy reinforcement learning techniques, primarily Proximal Policy Optimization (PPO), while also exploring other methods like Advantage Actor-Critic (A2C). 
In order to obtain real-time game state data, we plan to apply template matching from OpenCV. This will be a lightweight and efficient method of detecting the disctict and static sprites in Doodle Jump.

## Evaluation Plam 
**Quantitative Evaluation:** 
- Success will be measured quantitatively by comparing the agent’s performance to baseline strategies. The primary quantitative metrics will include average score and survival time. Baseline agents will include a random action agent and a simple heuristic agent which always moves towards the nearest platform. Each agent will be evaluated across multiple trials and learning curves will be analyzed. We expect our agent to perform at least 2x better than the baseline agents in score.
**Qualitative Evaluation:** 
- We will visualize gameplay rollouts to inspect the agent’s behavior, verifying that it makes purposeful movements and avoids falling unnecessarily. Debugging will include monitoring reward signals and analyzing failure cases to ensure that the decisions of the agent align with expectations. A successful result would demonstrate smooth gameplay with predictable outcomes and movement patterns that improve consistently. 


## AI Tool Usage 
We plan to use AI tools to brainstorm ideas for implementing RL concepts specific to Doodle Jump and assistance in debugging the code. All content generated from an AI will be reviewed in depth and modified as needed. Any use of AI tools will be fully documented to comply with course guidelines.
