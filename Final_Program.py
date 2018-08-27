import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
# import pickle

# Choose number of colours
size = 1000


# Choose the number of time you wish to run the algorithm
runs = 1

# Choose solution type
# 0 = returns a display of the colours in order given, no parameters apply
# 1 = Random starts, num of loops = num of random starting points
# 2 = Stochastic Hill Climb, num of iterations
# 3 = Iterated Local Search, num of loops = num of pertubations, num of iterations 
# 4 = Genetic Algorithm, pop_size, generations, parent_pool_size, Pm = Probability of Mutation, Ps is used for Rank Selection control
question_number = 2

       
#Parameters For questions 1-3                           
number_of_loops = 0
number_of_iterations = 1000000


#Parameters For Question 4, Genetic Algorithm
pop_size = 10 
generations = 100
parent_pool_size = 6
Pm = 0.8
Ps = 0.6

# Choose if you would like a fitness against iterations/generations line graph (True/False)
# if you have selected more than 5 runs you will also be given a BoxPlot of best Fitness
graphing = True

# Choose if you would like counters on (True/False) and how frequently, this will display the progress of the algorithm
# frequency for Q1 is for loops, for Q2,Q3 it is for iterations, for Q4 it is for generations
counters = True
frequency = 10




# Generating colors from file
def GetColours():
    colours =[]
    data = np.loadtxt("colours.txt", dtype= float, comments = '#',delimiter = ' ', skiprows = 4)
    for i in range(0, size):
        colours.append (data[i])
    colours = [arr.tolist() for arr in colours]
    return colours
 

# Taking two random points in the permutation, swapping them and reversing the permutation incuded    
def Swap_And_Reverse(permutation):                                          # Explanation in report
    start,end = sorted(random.sample(range(0,size),2))
    numbers_to_reverse = permutation[start:end+1]
    reversed_numbers_permutation = permutation[0:start]
    reversed_numbers_permutation[start:end+1] = numbers_to_reverse[::-1]
    reversed_numbers_permutation[end+1:size-1] = permutation[end+1:size]
    return reversed_numbers_permutation,start,end

# Two methods for calculting fitness,
# Swap_Fitness is an optimization that focuses only on the difference after the swap and reverse
# Fitness calculates distance between all colours in the permutation

def Swap_Fitness(permutation,start,end):
    l1dist =0
    l2dist = 0
    a = start-1
    b = start
    c = end
    d = end+1
    if start == 0:                              # Creating special cases for calculaing fitness
        a = b = start                           # is either the start or end of the swap are the first or last entries
    if end == size-1:                           # or are consecutive these must be compensated for
        c = d = end  
    for i in range(a,b):                    
        for j in range (3):                     # Calculating the Euclidean distance
            l1dist += (colours[permutation[i]][j]-colours[permutation[i+1]][j])**2
        l2dist += math.sqrt(l1dist)
        l1dist = 0
    for i in range(c,d):
        for j in range (3):
            l1dist += (colours[permutation[i]][j]-colours[permutation[i+1]][j])**2
        l2dist += math.sqrt(l1dist)
        l1dist = 0
 
    return l2dist

  
def Fitness(permutation):
    best_fitness = 0
    for i in range(size-1):
        distances = 0
        for j in range(3):                      # Calcuates the Euclidean distance between all points
            distances +=(colours[permutation[i]][j] - colours[permutation[i+1]][j])**2
        best_fitness += math.sqrt(distances)
    return best_fitness


# Pertubation for the ILS, this is two swap and reverses applied concurrently       
def Pertubation(permutation,p):
    if p == 1:
        start, end = sorted(random.sample(range(0,size),2))     # Applies two consecutive swap_and_reverse 
        entries_for_pertubation = random.sample(range(start,end+1),end+1-start)
        pertubated_permutation = permutation[0:start]
        for i in range(len(entries_for_pertubation)):
            pertubated_permutation.append(permutation[entries_for_pertubation[i]])
        pertubated_permutation[end+1:size]= permutation[end+1:size]   
    if p == 2:
        for _ in range(2):
            pertubated_permutation,start,end = Swap_And_Reverse(permutation)
    return pertubated_permutation, start, end

# Functions for genetic Algorithm
def Initial_Population_With_Fitness(pop_size):
    population =[]
    for _ in range(pop_size):                       # creates a population of random solutions of pop_size
        chromozone_and_fitness = []
        chromozone = random.sample(range(0,size),size)
        chromozone_and_fitness.append(chromozone)
        chromozone_and_fitness.append(Fitness(chromozone))      # Appends the fitness to each solution in the list to make ordered paird
        population.append(chromozone_and_fitness)
    population.sort(key=lambda elm : elm[1])                    # Sorts the population due to fitness
    return population

# Rank selection for selecting parents from parent pool, explained in report
def Rank_Selection(population,parent_pool,parent_pool_size,Ps):
    parents =[]
    p = 0
    while p < len(parent_pool):                     # Creates 2 parents from parent_selection_pool using rank selection
        if p < len(parent_pool) -1:                 # with probability for selecting first candidate = Ps
            if random.random() < Ps:
                parents.append(population[p])
                p = 0
                if len(parents) == 2:
                    return parents
        elif p == len(parent_pool) - 1:
            parents.append(population[parent_pool[-1]])
            p = 0
            if len(parents) == 2:
                return parents
        p +=1
        
# Creates a pool of potential parents and  and supplies these to rank selection function      
def Create_Parents(population,pop_size,parent_pool_size): 
    parent_pool = []
    parent_pool.extend(sorted(random.sample(range(0,pop_size),parent_pool_size)))
    parents = Rank_Selection(population,parent_pool,parent_pool_size,Ps)
    return parents

# Creates children from 2 selected parents
def Create_Children(parents):
    children = []
    crossover_point = random.randint(0,size-1)          # Creates random point for crossover, n
    child0 = parents[0][0][0:crossover_point]           # First child receives first part of its chromosome from first parent, up to index n
    child1 = parents[1][0][0:crossover_point]           # Second child receives the first part of its chromosome from second parent, up to index n
    for gene in parents[1][0][crossover_point::]:       # First child receives second part of its chromosome from second parent sequencially, from index n on
        if gene not in set(child0):                     
            child0.append(gene)                         
    index = 0    
    while len(child0) < size:                           # First child receives last part of its chromosome from second parent up to index n
        if parents[1][0][index] not in set(child0):
            child0.append(parents[1][0][index])
        index += 1
    for gene in parents[0][0][crossover_point::]:
        if gene not in set(child1):                     # The rest of the code does the same for the second child, taking genes from first parent
            child1.append(gene)
    index = 0        
    while len(child1) < size:                
        if parents[0][0][index] not in set(child1):
                child1.append(parents[0][0][index])
        index += 1
    children.extend([child0,child1])
    return children        
            
# Applies permutation, possibly, to the children            
def Mutation(permutation,Pm):            
    mutated_children =[]
    child0 = permutation[0]
    child1 = permutation[1]
    mutation = random.random()  
    if mutation < Pm:                                   # If random number generated < Pm, applies swap and reverse to first child
        child0,s,e = Swap_And_Reverse(child0)
    P_mutation = random.random()
    if P_mutation < Pm:                                 # As above for second child
        child1,s,e = Swap_And_Reverse(child1)
    mutated_children.extend([child0,child1])
    mutated_children[0] = [mutated_children[0],Fitness(mutated_children[0])] 
    mutated_children[1] = [mutated_children[1],Fitness(mutated_children[1])]  
    return mutated_children



# Main pertubation generator
def Solution(q,number_of_loops,number_of_iterations,pop_size,generations,parent_pool_size,Pm,Ps):
    initial_permutation = list(np.arange(size))
    best_solution = []
    best_fitness = Fitness(initial_permutation) 

    if q == 0:
        initial_permutation = list(np.arange(size))     # This creates the colours in the same order as given in the .txt
        best_solution = initial_permutation             # it was used for comparason during writing
        best_fitness = Fitness(initial_permutation)
        x = []
        y = []

    # Random_starts 
    elif q == 1:
        random_solution = random.sample(range(0,size),size)         # cretaes a random order of the colours
        best_fitness = Fitness(random_solution)
        x = [0]
        y = [best_fitness]
        for loop in range(number_of_loops):                         # generates a random number of starting points, defined by_number_of_loops
            random_solution2 = random.sample(range(0,size),size)
            if Fitness(random_solution2) < best_fitness:
                best_solution = random_solution2                    # Keeps track of best_fitness and its corresponding permutation
                best_fitness = Fitness(random_solution2)
                x.append(loop)                                      # x and y are trackers for line graph of fitness against loops
                y.append(best_fitness)
                if counters ==True:                                 # Used for counters
                    if loop%frequency == 0:
                        print "run",run,"loop",loop
            else:
                best_solution = random_solution
                
    # Hill_climbing         
    elif q == 2:     
        random_solution = random.sample(range(0,size),size)             # Creates a random starting point
        best_solution = random_solution
        best_fitness = Fitness(random_solution)
        x = [0]
        y = [best_fitness]
        for iteration in range(number_of_iterations):                            # Starts hill climber, 
            first_improvement,start,end = Swap_And_Reverse(random_solution)     # Applies inversions
            if counters == True:                                                # used for counters
                    if iteration%frequency == 0:  
                        print "run",run,"iteration",iteration  
            if Swap_Fitness(first_improvement,start,end) < Swap_Fitness(random_solution,start,end):     # Updates our current position if step reduced fitness
                random_solution = first_improvement                         
                best_fitness = Fitness(first_improvement)               # Keeps track of best_fitness and it's corresponding permutation
                best_solution = random_solution
                x.append(iteration)                                     # x and y used to create line graphs
                y.append(best_fitness)
    # Iterated_local_search    
    elif q == 3:
        x = [0]
        y = [best_fitness]
        initial_solution = random.sample(range(0,size),size)            # creates random starting point
        for loop in range (number_of_loops):                            # sets pertubation loop
            random_solution,start,end = Pertubation(initial_solution,2) # applies pertubation, 2*swap_and_reverse
            for iteration in range (number_of_iterations):              # sets hill climbing within the ILS
                reversed_numbers,start,end = Swap_And_Reverse(random_solution)  #rest of this operates as the Hill CLimber
                if counters == True:                                    # counters
                    if iteration%frequency == 0:
                        print "run",run,"loop",loop,"iteration", number_of_iterations*loop+iteration
                if Swap_Fitness(reversed_numbers,start,end) < Swap_Fitness(random_solution,start,end):
                    random_solution = reversed_numbers                  # Keeping track
            initial_solution = reversed_numbers
            if Fitness(reversed_numbers) < best_fitness:
                best_fitness = Fitness(reversed_numbers)                # Saving best_fitness before another pertubation is applied
                best_solution = reversed_numbers
                x.append(iteration + loop*iteration)                    # x and y again used for line graphs
                y.append(best_fitness)
       
    # Genetic_Algorithm     
    elif q == 4:
        x =[]
        y= []
        population = Initial_Population_With_Fitness(pop_size)          # Calls function to set inital population
        best_fitness = population[0][1]
        for gen in range(generations):
            candidates_for_new_gen = []
            parents = Create_Parents(population,pop_size,parent_pool_size)  # Calls function to create parents
            children = Create_Children(parents)                             # Creates children from these parents
            children = Mutation(children,Pm)                                # Mutates children, possibly
            candidates_for_new_gen.extend(parents)                          # Adds the parents and children to a sub group
            candidates_for_new_gen.extend(children)   
            candidates_for_new_gen.sort(key=lambda elm : elm[1])            # Sorts this sub group due to fitness
            population[pop_size-1] = candidates_for_new_gen[0]
            population[pop_size-2] = candidates_for_new_gen[1]              # Takes the two chromosomes in this sub group with the lowest fitness
            population.sort(key=lambda elm : elm[1])                        # they replace the two worst in the population, then sorts population again
            if counters == True:                                            # Counters
                if gen%frequency == 0:
                    print "run",run +1, "gen", gen 
            if population[0][1] < best_fitness:                             # Keeps track of best fitness and corresponding chromosome
                best_fitness = population[0][1]
                best_solution = population[0][0]
                x.append(gen)
                y.append(best_fitness)                                      # x and y used for line graphs 
    return best_solution,best_fitness,x,y




    
# Display the colours in the order of the permutation in a pyplot window 
def plot_colours(colours,permutation):
    assert len(colours) == len(permutation)
    ratio = 50 # ratio of line height/width, e.g. colour lines will have height 10 and width 1
    img = np.zeros((ratio, len(colours), 3))
    for i in range(0, len(colours)):
        img[:, i, :] = colours[permutation[i]]
    fig, axes = plt.subplots(1, figsize=(8,4)) # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.title(str(size)+" Colours      Fitness = " +str(best_ever_fitness)+"\n"+Label)
    plt.show()
    
def graph():                        # Creates line graphs of fitness against iterations or generations
    for i in range (runs-1):        
        plt.plot(iterations_list[i],fitness_list[i])     
    plt.plot(iterations_list[runs-1],fitness_list[runs-1])   
    Label2 = "Iterations"
    if question_number == 4:
        Label2 = "Generations"
    plt.title(Label+ "\n"+ str(size)+"   colours" + "    mean = " +str(np.mean(best_fitness_list)) + "    std =  " +str(np.std(best_fitness_list)))
    plt.xlabel(Label2)
    plt.ylabel("Fitness")
    plt.show()
    
    if runs > 5:                    # If doing more than 5 runs creates box plots also
        plt.boxplot(best_fitness_list)
        plt.ylabel(' Best Fitness')
        plt.xticks([1],[Label])
        plt.title(str(size) +"  colours ")
        plt.show()
    
def label():                        # Creates labels for graphs due to Algorithm chosen and set parameters
    if question_number == 0:
        Label = "Original order of colours"
    if question_number == 1:
        Label ="Random Start Point"
    elif question_number == 2:
        Label = "Hill Climb"
    elif question_number == 3:
        Label = "Iterated Local Search"
    elif question_number == 4:
        Label = "Genetic Algorithm  Generations = "+str(generations)+"    Population = " +str(pop_size) + "    Pm = " +str(Pm)
    return Label
    
    
    
    
  
colours = GetColours()
permutations = []
best_fitness_list = []
iterations_list =[]
fitness_list =[]
t = time.time()
for run in range(runs):
    permutation_,best_fitness,iterations,fitness = Solution(question_number,number_of_loops,number_of_iterations,pop_size,generations,parent_pool_size,Pm,Ps)
    permutations.append(permutation_)
    best_fitness_list.append(best_fitness)
    iterations_list.append(iterations)
    fitness_list.append(fitness)
total_time = time.time() - t
best_ever_fitness = best_fitness_list[np.argmin(best_fitness_list)]    
Label = label()


print size, "  Colours   ",Label   
print "Best Fitness = " ,best_ever_fitness
if runs > 1:
    print "Average Best Fitness = " ,np.mean(best_fitness_list)
    print "Standard Deviation = " ,np.std(best_fitness_list)
print "Time = ", total_time, " seconds"

permutation = permutations[np.argmin(best_fitness_list)]
plot_colours(colours,permutation)
if graphing == True:
    graph()



