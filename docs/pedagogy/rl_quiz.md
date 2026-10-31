# RL understanding quiz (answer ONLY from what the course material teaches)

Answer each question in 2-5 sentences. If the material you read does not teach
something needed to answer, write "NOT COVERED" rather than using outside knowledge.

Q1. In this project's self-play loop, what plays the role of the agent, the
environment, the state, the action and the reward? When does the agent receive reward?

Q2. Module 4 trains the network on human games. Is that reinforcement learning?
What is it called, and what is its fundamental ceiling?

Q3. AlphaZero is often described as "policy evaluation + policy improvement".
In this codebase, which component performs policy improvement, which performs
policy evaluation, and how does the improved policy get back into the network?

Q4. Each self-play position's value target is the final game result. Is that a
Monte Carlo target or a temporal-difference (bootstrapped) target? Give one
advantage and one disadvantage of that choice.

Q5. Name every exploration mechanism in this project's self-play pipeline and say
what level each one operates at (inside the search, at move selection, ...).

Q6. Someone sets the replay buffer to keep every game forever, starting from
iteration 1. Predict what goes wrong, and explain it in RL terms.

Q7. In a self-play game, the eventual winner blundered on move 20 but the opponent
failed to punish it. What value target does that move-20 position receive? What is
the name of the underlying problem, and why does the algorithm tolerate it?

Q8. Does this project's network learn action values Q(s,a)? If not, what does it
learn, and where do action values come from instead?

Q9. You want to apply this exact loop to Connect Four. Which parts of the codebase
would have to change, and which would carry over unchanged?

Q10. Why does the loop include an arena gate instead of always using the newest
network? Answer in terms of what can go wrong in RL with function approximation.

Q11. Before running a self-play experiment at the course's CPU scale, what does the
material tell you to expect to see, and how would you tell real learning from noise?

STUCK LOG (required, separate section): list every point in the material where
you (a) did not know what to do next, (b) could not tell what a step was for or how
it fit the bigger picture, (c) would not know how to run a step as written, or (d)
had no way to tell whether your result was right. Cite the file and step for each.
