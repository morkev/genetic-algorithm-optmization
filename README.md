# Optmization - Genetic Algorithms

Today I came across a video about optimization and genetic algorithms, published by Ahmad Hably [here](https://www.youtube.com/watch?v=x-eqUoCIXi8). After taking some notes, I was delighted and thought it would be interesting to create an implementation of a genetic algorithm to solve the knapsack problem.

What did I code? Well, I am using a genetic algorithm to solve the Knapsack problem with a specific limit or capacity C and a given set of N objects. The Knapsack contains objects with values and weights, and we seek to maximize profit while staying under the space limit. Our fitness function sums up the profits of the objects in the Knapsack.


## Theoretical Overview
Rather than exploring a single starting point, exploration is initialized by a set of randomly generated potential solutions - these are known as deign points or population.
The population of solutions, at each iteration, are updated into a new population through the application of three operators: <b>Selection</b>, <b>Cross_Over</b> and <b>Mutation</b>.

| Selection | Cross_Over & Mutation |
| ------------- | ------------- |
| Exploits the current population of "parents" to retain a "mating pool". The most "promising" designs will likely lead to an optimal design in future generations.  | Explore the design space, creating new (hopefully) better designs in the "children" population which will become the parents of the new generation. | 

<b>NOTE</b>: A fitness value is associated to each solution and is related to the objective function.

### Comparing Genetic Algorithms to Local Methods
1. Genetic algorithms use a coding of the parameters and not the parameters themselves.
2. Genetic algorithms work with a population of designs.
3. Genetic algorithms make use of the values of the objective function only and do not need the gradient (gradient-free method).
4. It relies on probabilistic and not deterministic transition rules.
5. Genetic algorithms are less sensitive to computational noise - errors on the objective function value that can translate into an error on the fitness function.


## Mathematical Overview
$max\ f(x)$ : $x \ \epsilon \ [x_{min}, \ x_{max}]$
- Without loss of generality: binary encoded case.
- Variable is coded as a binary string of fixed length.

```r
0, 1, 0, ... , 1, 1 chromosome
each cell is a "gene"

- The lower bound is coded to 0 0 0 _ _ _ 0
- The upper bound is coded to 1 1 1 _ _ _ 1
```

The length of the chromosome (binary string) is determined from the targeted accuracy on the $x$ variable. An initial population is randomly generated from a stochastic building of 0s and 1s sequence of length ($l$). Finally, the real $x$ associated to each of the chromosomes is then computed:
- $x=x_{min} + \frac{{x_{max} \ -} \ {x_{min}}}{2^l \ - \ 1} \ E$
- $with\ E\ = \sum_{i=0}^{l-1} \ b{_i} 2^i$

A fitness function (related to the objective function) can now be computed, and will be used to perform the "selection" process.

### Maximization Problem
Fitness can be taken equal to the objective function.
- Or the square of it ($N^2$).
- Or square root of the objective function ($\sqrt{N}$).

### Minimization Problem
$\frac{1}{1+f(x)}$ or $C-f(x)$ with $C\geq\max(f(x))$

So that a minimum value of the objective function corresponds to a maximum value of the fitness function.



