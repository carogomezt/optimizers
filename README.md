# Optimizers
Some methods to optimize search.

## Tabu Search
This is an implementation of the Tabu Search for the Traveling Salesman Problem
using the dataset of  [National Traveling Salesman Problems](http://www.math.uwaterloo.ca/tsp/world/countries.html#QA).

The steps to run this implementation are:

1. Clone this repository

2. Download in the same folder the points of the cities that you want to visit:
i.e. [Points of Western Sahara](http://www.math.uwaterloo.ca/tsp/world/wi29.tsp)
   
3. Run the code with the following command:

    `python tabu.py -f <name_file> -i <max_iters>`

    *name_file:* In the case of Western Sahara it would be _wi29.tsp_

    *max_iters:* It is recommended to play with this value, init with 200.