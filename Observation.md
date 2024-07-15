- Position cost
  - Allow backtracking because it only cares about position
  - It forces the calculated velocity to match the spacing of the global plan (fixed delta t)
- Velocity ref
  - Tell the robot to follow the global plan at a certain speed
  - Might not allow backtracking because the ref velocity is positve
- Acceleration
  - Smooth velocity change
  - Might not stop fast enough to avoid collision
- Theta error

- omega W

  - Make the lidar and the map misalign
  - But limit the curvature of the turn
    - It makes the robot act like an ackermann
    - Trade-off

- It is better to make the global planner slow

- Decrease acceleration is at odd with following the position of the trajectory

- Sometimes, putting the maximum hinder the solver from finding a better solution

  - Trade off between the "obvious" fast solution and safety

- Constraining the last horizon step

  - Might result in the bending of the mpc plan if it wants to match a certain velocity reference, because going off course will give higher velocity

- Add time as an optimization variable

  - What would it do? Make the give more variance to the velocity?

- Tips:

  - Increase inflation to 0.31
  - Increase publish to 10.0
  - Make the map obstacle beyond the blind spot

- weight acceleration, make the movement smoother, rejecting quick changes in the planning

- Remove the angle theta for backtracking
- The more value you initialize, the less N you can reduce
- For better backtrack, remove the horizon final constraint
- More N increase checking, but it also smooth out the path
  ========= Good Check point!
- w = 0.8 seems to be safe? It takes time to reverse but still safe in many situation, prevent excessive rotation

- There are times when it behaves wildly?!
- Short horizon tell the robot to turn in toward the correct orientation first

- Time elastic stop the robot from vary the velocity given the same position, less change in time leads to less change in velocity
