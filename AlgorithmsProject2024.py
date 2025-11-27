

import numpy as np
import random
#from numpy.random import random
import time

random_num = np.random.randint(1,50)
N = random_num
matrix = np.random.randint(-1000, 1000, (N, N))  # Creates an N x N matrix with random integers from -1000 to 1000
print("The current matrix is of size :" +str(random_num) + "X" +str(random_num) + "\n" + str(matrix))
###################################################################################
OriginalPaths = []
###########################################################################################
def printPath(path) :# a function that prints all the elements in the path
 print("path : ")
 for i in path :
  print(f"{i} , ", end=" ") #for each element 'i' in path, print it
 print()

###########################################################################################
def TotalcostPaths(OriginalPath) :
 totalCostLists = [] #a list that stores the total cost list  for each path
 for counter,i in enumerate(OriginalPath) :
    totalCost = 0
    totalCost_List = [] # a list that stores the current total cost path list for the current path

    for j in i :
      totalCost += j #calculates the current cost of cell in path 'i'
      totalCost_List.append(totalCost) #appends the current total cost to the list
    totalCostLists.append(totalCost_List) # append and add the current total list to the totalCostList list
 for counter, cost in enumerate(totalCostLists) :
   print(f"TotalCost{counter} : {cost}")  # print the current total cost list

 return totalCostLists
###########################################################################################
def leastCostPaths(TotalcostPaths) : #after calculating the total cost for each path, this is a function that returns the least total cost list path
 # in addition to the minimum total cost
 minimum = float("inf")  # the variable that stores the minimum total cost
 leastCostList = [] #the list that stores the minimum total cost path list
 for i in TotalcostPaths : # a loop that checks the least total cost path by checking the last element in the list
  # (because the last element in the list represent the total cost of the current path)
    last_element = i[-1]
    if last_element < minimum: #comparison for computing the minimum total cost
     minimum = last_element
     leastCostList=[i] # 'i' represents the list
    elif last_element == minimum: #
       leastCostList.append(i) #add to the list if there is another least total cost path
 print("The total minimum cost path of the matrix is: "+str(minimum))
 return leastCostList
###########################################################################################
def leastweightedPath(OriginalPath) : #this is a function that returns the original least cost path without adding previous elements
 # and computing the total cost
 minimum = float("inf")  #calculates the minimum total cost
 leastWeightedPathList = []
 for i in OriginalPath : #this loop is for calculating the total cost for each path
  # and storing the minimum total cost between all paths
  # and storing the original minimum cost path
  total = sum(i) # the total sum of this current path
  if total < minimum:
   minimum = total
   leastWeightedPathList=[i]
  elif total == minimum:
   leastWeightedPathList.append(i)
 return leastWeightedPathList
###########################################################################################


def AllPossiblePaths(matrix, path , i, j, M, N, visited ): #a function that shows all possible paths from source to destination
# i---> rows,  j--->columns,  M--->first Element in the matrix, N---> Last element in the matrix ,
# visited--->boolean list that checks if the current element is visited or not to avoid redundancy and infinite loops
#matrix--->input matrix, path---> a list that stores each element from source to destination
 if i == M-1 and j == N-1: #checks the boundaries of the matrix , if the last element (Destination) is reached, then print the matrix
  path.append(matrix[i][j]) # append the first path to the list 'path'
  printPath(path) #call the function 'printPath' to print this current path
  OriginalPaths.append([int(x) for x in path]) #OriginalPaths is a 2D list that stores all possible paths from source to destination
  path.pop() #Backtrack, this helps in finding other possible paths
  return

 if i < 0 or  i >= M or j<0 or j >= N or visited[i][j]  : #checks if the current position is out of bound or visited,if true, it will exit the function
  return

 path.append(matrix[i][j]) #if previous statement is false appending to the path list will execute
 visited[i][j] = True

 AllPossiblePaths(matrix, path, i + 1, j, M, N,visited) # recursion call move down
 AllPossiblePaths(matrix, path, i, j + 1, M, N, visited)# recursion call move right
 AllPossiblePaths(matrix, path, i - 1, j, M, N,visited)# recursion call move up
 AllPossiblePaths(matrix, path, i, j - 1, M, N, visited)# recursion call move left

 path.pop() #this allows exploring other paths
 visited[i][j] = False


####################################Rimas###############################################

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))

    def bellman_ford(self, src, dest):
        dist = [float('inf')] * self.V
        dist[src] = 0

        for _ in range(self.V - 1):
            for u, v, weight in self.edges:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

        return dist[dest]

class ProjectGraphBased:
    def __init__(self, N):
        self.N = len(N)
        self.matrix = matrix

    def fill_arr(self):
        self.matrix = [[random.randint(-1000, 1000) for _ in range(self.N)] for _ in range(self.N)]

    def graph_based_algorithm(self):
        start = time.time()
        self.min_bellman()
        end = time.time()
        execution_time = (end - start) * 1000  # Convert to milliseconds
        print(f"The Execution Time for Minimum path By using Bellman-Ford algorithm for input size {self.N} = {execution_time:.2f} ms")

    def min_bellman(self):
        rows, cols = self.N, self.N
        V = rows * cols
        graph = Graph(V)

        for i in range(rows):
            for j in range(cols):
                node = i * cols + j
                if i > 0:
                    graph.add_edge(node, node - cols, self.matrix[i - 1][j])
                if j > 0:
                    graph.add_edge(node, node - 1, self.matrix[i][j - 1])
                if i < rows - 1:
                    graph.add_edge(node, node + cols, self.matrix[i + 1][j])
                if j < cols - 1:
                    graph.add_edge(node, node + 1, self.matrix[i][j + 1])

        source = 0  # Starting node (0, 0)
        destination = rows * cols - 1  # Ending node (n-1, n-1)
        minPath = graph.bellman_ford(source, destination)
        print(f"The Minimum Total Path Value: {minPath}")
print("GRAPH-BASED ALGORITHM")
project = ProjectGraphBased(matrix)
project.graph_based_algorithm()

###################################################################################



#####################################Sara#############################################

def min_path_sum(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
 # Initialize a DP table to store the minimum path sum to each cell
    dp_table = [[0] * cols for _ in range(rows)]
  # Initialize the first cell
    dp_table[0][0] = matrix[0][0]
    # Fill the first row
    for col in range(1, cols):
       dp_table[0][col] = dp_table[0][col - 1] + matrix[0][col]
   # Fill the first column
    for row in range(1, rows):
       dp_table[row][0] = dp_table[row - 1][0] + matrix[row][0]
   # Fill the remaining cells
    for row in range(1, rows):
       for col in range(1, cols):
           dp_table[row][col] = min(dp_table[row - 1][col], dp_table[row][col - 1]) + matrix[row][col]
    return dp_table[rows - 1][cols - 1]





def get_least_weight_path(matrix, dp_table):
    rows = len(matrix)
    cols = len(matrix[0])

  # Initialize path starting from the bottom-right corner
    path = [(rows - 1, cols - 1)]

    row = rows - 1
    col = cols - 1

  # Trace back the path from the bottom-right corner to the top-left corner
    while row > 0 or col > 0:
            if row == 0:  # We can only move left
              col -= 1
            elif col == 0:  # We can only move up
             row -= 1
            else:
                return
         # Move in the direction of the minimum cost
            if dp_table[row - 1][col] < dp_table[row][col - 1]:
              row -= 1
            else:
             col -= 1
            path.append((row, col))

  # Reverse the path to get the correct order from top-left to bottom-right
    path.reverse()

    return path



print("DYNAMIC PROGRAMMING ALGORITHM")
  # Calculate the DP table using the dynamic programming method
dp_table = min_path_sum(matrix)

  # Calculate the least weight path (coordinates)
least_weight_path = get_least_weight_path(matrix, dp_table)

  # Print the results
print("The least weight path is:", least_weight_path)

result = min_path_sum(matrix)
print("Minimum path sum:", result)

########################################################################################
print("BRUTE-FORCE ALGORITHM")

def BruteForce(matrix):
 row = len(matrix)  # number of rows
 column = len(matrix[0])
 path = []
 visited = [[False for _ in range(column)] for _ in range(row)]
 AllPossiblePaths(matrix, path , 0, 0, row, column,visited)
 print("This is a trial to see original path" + str(OriginalPaths))
 total_costs = TotalcostPaths(OriginalPaths)
 least_cost_path = leastCostPaths(total_costs)
 least_weighted_Path = leastweightedPath(OriginalPaths)
 print("Total cost for all paths: "+str(total_costs))
 print("The least Original path cost is :  " + str(least_weighted_Path))
 print("the least Total cost path is :" +str(least_cost_path))

BruteForce(matrix)











