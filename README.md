# policy-random-hill-climb-on-cart-pole
Randomized Hill Climb to search for optimal policy parameters in CartPole

Philip Docena

When I first read about reinforcement learning years ago, the first concept I learned was what I came to know as Q-learning, which calculates actions given some states via a comparison of maximized action-values.  It has proven its worth over the years (in recent years, DQN).  Less well-known (at least to me before OMSCS) is the concept of learning the policy directly from trajectories.

In the context of the type of problems encountered in RL (modeled by MDPs), a direct policy parameter search is an optimization problem.  This can be solved with standard randomized optimization algorithms like hill climbs, simulated annealing, and genetic algorithms (and stochastic gradient descent).  The presented code is for a randomized hill climb (RHC), for which I explore a simple application.

The cart pole (Cartpole-v0 and Cartpole-v1) environment in OpenAI Gym is quite simple.  A pole-carrying cart moves along a line in a 2D plane, with the goal of keeping the pole standing within a few degrees of vertical for as long as possible.  The pole-cart hinge is frictionless, which simplifies some of the world physics.  The allowed actions are to move the cart left or right (binary force), while keeping the pole within a tilt limit (15 degrees) and the cart within some bounds (i.e., do not bump the edge of the world).  The system provides four sensor readings (tilt, angular momentum, location, velocity).  Each episode is also terminated after a certain number of time steps, 200 for v0 and 500 for v1.  This should be a simple control problem.  Intuitively, it is also simple mechanically: if the pole is leaning left, move left; leaning right, move right.

RHC should be well-suited to CartPole.  From the above intuition, the policy surface appears to be simple, although the effect of varying angular momentum would make the policy surface complex at small tilt angles.  For example, if the pole is tilting to the left (and the intuitive action is to move left to keep the pole upright), but the angular momentum is to clockwise, should the cart move to the left per our intuition?  The optimization must figure this out.

In practice, as I have observed in this example, RHC can be difficult to run correctly on continuous spaces.  The amount of the epsilon change for the parameters is domain dependent (so domain knowledge is relevant), and small variations in starting conditions (e.g., random seeds, initial weights, initial step size, step size decay rate) produce wildly different convergence speed.  Sometimes it fails to converge after a reasonable number of iterations, if the step size keeps on overshooting an optima, or the number of attempts results in a random number sequence that keeps on missing a good solution.  RHC is also best used on static problems where the objective function calculation stays the same from trial to trial.  This is not true in Cartpole since the initial conditions vary, so a parameter that might have led to an ascent (and hence the parameter change is made permanent) might turn out to be a descent in the following trial.  However, when well-suited to a problem, the primary advantage of RHC is its simplicity and speed, particularly when the scoring function is low cost. -- PD

