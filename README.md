# Operations_Final_Submission
Final Submission of the Operations Optimization Assignment

This repository contains copied and pasted files from the repository developed by THEO. 
I copied the simplest files, in order to get the algorithm working. 
We will need to dive into the workings and if it is implemented correctly. 
For now I will focus on plotting the results for verification. 
#+#+# Gabo
Esta es mi branch. 


# Explanation
This program solves an optimization problem of container allocation, between different barges and trucks, to move them between different port terminals.
The algorithm that solves this allocation, is split into:
- An initial greedy algorithm that finds a feasible solution.
- A Meta-Heuristic algorithm that improves that feasible solution. 


The process consists of the following steps:
- 1. Generate initial scenario. This requires stating the barge capacities, barge fixed costs, container handling time, and per container trucking cost (specific to the two container sizes). THIS IS CARRIED OUT IN THE INITIALIZER OF THE GreedyOptimizer CLASS.

- 2. Then, the .generate_instance() method, creates the specific random scenario. So it produces, using a seed, the number of containers, the number of terminals, the travel time matrix (NxN) between specific terminals, and for each container a dictionary with its properties (opening time, closing time, weight, import or export, and assigned terminal.).

- 3. Then, the Greedy algorithm solves the problem,   using some imported module that solves the traveling salesman problem,   obtaining an initial feasible solution. #[I think the outputs of get_route, and Barge routing matrix xijk might be very useful for verification.] This solution is obtained in seconds, and is printed out to the terminal.


- 4. Then, the Meta-Heuristic algorithm does some magic and improves it. -> Figure out what is doing. 


- 5. Then, a sensitivity analysis is performed.