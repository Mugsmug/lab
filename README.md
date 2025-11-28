1. Implementation of Linear Search Algorithm

def linearsearch(arr,key):
    for i in range(len(arr)):
        if arr[i] == key :
            return i
    return -1

arr=list(map(int,input("Enter the elements --> ").split()))
key=int(input("Enter the element to be search : "))

result = linearsearch(arr,key)

if result != -1 :
    print(f"Element found at index {result} is {key}")
else :
    print("Element not found")
________________________________________
2. Implementation of Binary Search Algorithm
   
def binarysearch(arr,key):
    low = 0
    high = len(arr)-1
    
    while low <= high :
        mid = low + (high - low)//2
        
        if arr[mid] == key:
            return mid
        elif arr[mid] < key :
            low = mid + 1
        else :
            high = mid - 1
    return False

arr=[]
arr=list(map(int,input("Enter the elements in the array : ").split()))
arr.sort()
print(arr)
key=int(input("Enter the element to search --> "))
result = binarysearch(arr,key)

if result != False:
    print(f"Element found at index {result} is {key}")
else :
    print("Not Found")

________________________________________
3. Implementation of Merge Sort
Question: Implement Merge Sort using the divide and conquer approach.

def mergesort(arr):
    if len(arr)<=1:
        return arr
    mid = len(arr)//2
    lefthalf=arr[:mid]
    righthalf=arr[mid:]
    
    sortedhalf=mergesort(lefthalf)
    sortedright=mergesort(righthalf)
    
    return merge(sortedhalf,sortedright)
def merge(left,right):
    result=[]
    i=j=0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i = i + 1
        else :
            result.append(right[j])
            j = j + 1
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

arr=list(map(int,input("Enter the elements in the array --> ").split()))
print("Original array is --> ",arr)
sortedarr=mergesort(arr)
print("Sorted array will be ",sortedarr)

________________________________________
4. Implementation of Quick Sort (Divide and Conquer)
Question: Implement Quick Sort using divide and conquer.

def quicksort(arr):
    if len(arr) <=1:
        return arr
    
    pivot=arr[-1]
    less=[]
    greater=[]
    equal=[]
    
    for x in arr:
        if x < pivot :
            less.append(x)
        elif x > pivot :
            greater.append(x)
        else:
            equal.append(x)
    
    return quicksort(less) + equal + quicksort(greater)

arr=list(map(int,input("Enter the elements in the array --> ").split()))
print("Original array is -- > ",arr)

result=quicksort(arr)
print("Sorted array will be --> ",result)
________________________________________
5. Implementation of Strassen’s Matrix Multiplication (Divide and Conquer)
Question: Multiply two matrices using Strassen’s algorithm.


def add_matrix(A, B):
    """Add two matrices."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def sub_matrix(A, B):
    """Subtract two matrices."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def split_matrix(A):
    """Split matrix into 4 submatrices."""
    n = len(A)
    mid = n // 2

    a11 = [row[:mid] for row in A[:mid]]
    a12 = [row[mid:] for row in A[:mid]]
    a21 = [row[:mid] for row in A[mid:]]
    a22 = [row[mid:] for row in A[mid:]]

    return a11, a12, a21, a22


def strassen(A, B):
    """Strassen's Matrix Multiplication (Divide & Conquer)."""

    n = len(A)

    # Base case: 1x1 matrix
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    # Split matrices into 4 halves
    a11, a12, a21, a22 = split_matrix(A)
    b11, b12, b21, b22 = split_matrix(B)

    # Compute Strassen's 7 products
    p1 = strassen(add_matrix(a11, a22), add_matrix(b11, b22))
    p2 = strassen(add_matrix(a21, a22), b11)
    p3 = strassen(a11, sub_matrix(b12, b22))
    p4 = strassen(a22, sub_matrix(b21, b11))
    p5 = strassen(add_matrix(a11, a12), b22)
    p6 = strassen(sub_matrix(a21, a11), add_matrix(b11, b12))
    p7 = strassen(sub_matrix(a12, a22), add_matrix(b21, b22))

    # Calculate result submatrices
    c11 = add_matrix(sub_matrix(add_matrix(p1, p4), p5), p7)
    c12 = add_matrix(p3, p5)
    c21 = add_matrix(p2, p4)
    c22 = add_matrix(sub_matrix(add_matrix(p1, p3), p2), p6)

    # Combine 4 results into one matrix
    new_matrix = []
    for i in range(n // 2):
        new_matrix.append(c11[i] + c12[i])
    for i in range(n // 2):
        new_matrix.append(c21[i] + c22[i])

    return new_matrix


def pad_matrix(A):
    """Pad matrix to nearest power of 2."""
    import math
    n = len(A)
    m = len(A[0])
    size = max(n, m)
    next_power = 2 ** math.ceil(math.log2(size))

    padded = [[0] * next_power for _ in range(next_power)]
    for i in range(n):
        for j in range(m):
            padded[i][j] = A[i][j]

    return padded, n, m


def unpad_matrix(C, rows, cols):
    """Remove padding to restore original matrix size."""
    return [row[:cols] for row in C[:rows]]


# ---------------- MAIN PROGRAM ---------------- #

print("Enter size of matrix:")
n = int(input())

print("\nEnter Matrix A:")
A = [list(map(int, input().split())) for _ in range(n)]

print("\nEnter Matrix B:")
B = [list(map(int, input().split())) for _ in range(n)]

# Pad matrices if needed
A_pad, rA, cA = pad_matrix(A)
B_pad, rB, cB = pad_matrix(B)

# Compute Strassen result
C_pad = strassen(A_pad, B_pad)

# Unpad to original size
C = unpad_matrix(C_pad, rA, cB)

print("\nResultant Matrix (Using Strassen's Divide & Conquer):")
for row in C:
    print(row)


________________________________________
6. Implementation of Knapsack using Greedy Method
Question: Solve fractional knapsack problem using greedy approach.


def fractional_knapsack(values, weights, capacity):
    n = len(values)
    ratio = [(values[i]/weights[i], weights[i], values[i], i) for i in range(n)]
    ratio.sort(reverse=True)
    
    total_value = 0
    taken = [0] * n
    
    for r, w, v, idx in ratio:
        if capacity == 0:
            break
        if w <= capacity:
            taken[idx] = 1
            total_value += v
            capacity -= w
        else:
            taken[idx] = capacity / w
            total_value += v * (capacity / w)
            capacity = 0
    
    return total_value, taken

def greedy_coloring(adj):
    n = len(adj)
    color = [-1] * n
    
    for u in range(n):
        used = set()
        for v in adj[u]:
            if color[v] != -1:
                used.add(color[v])
        
        c = 0
        while c in used:
            c += 1
        color[u] = c
    return color

def knapsack_menu():
    n = int(input("Enter number of items: "))
    values = []
    weights = []
    for i in range(n):
        v, w = map(float, input(f"Enter value and weight of item {i}: ").split())
        values.append(v)
        weights.append(w)
    capacity = float(input("Enter capacity of the knapsack: "))
    
    max_value, taken = fractional_knapsack(values, weights, capacity)
    print(f"\nMaximum value in knapsack = {max_value}")
    print("Items taken (fraction or 1 = full, 0 = not taken):")
    for i, t in enumerate(taken):
        print(f"Item {i}: {t}")

def coloring_menu():
    n = int(input("Enter number of vertices--> "))    
    adj = [[] for _ in range(n)]

    print("Enter number of edges. Enter -1 -1 to stop")
    print("Vertices should be from 0 to", n-1)

    while True:
        u, v = map(int, input().split())
        if u == -1 and v == -1:
            break
        if u < 0 or u >= n or v < 0 or v >= n:
            print("Invalid vertices!!!", n-1)
            continue
        adj[u].append(v)
        adj[v].append(u)

    colors = greedy_coloring(adj)
    print("Colors assigned to each vertex is --> ")
    for i, c in enumerate(colors):
        print(f"Color of vertex {i} is {c}")
    print("Total colors used--> ", max(colors)+1)

def main():
    while True:
        print("\nGreedy Algorithms Menu:")
        print("1. Fractional Knapsack")
        print("2. Graph Coloring")
        print("3. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            knapsack_menu()
        elif choice == '2':
            coloring_menu()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()

#Greedy Algorithms Menu:
1. Fractional Knapsack
2. Graph Coloring
3. Exit
Enter your choice: 2
Enter number of vertices--> 5
Enter number of edges. Enter -1 -1 to stop
Vertices should be from 0 to 4
0 1
0 2
1 2
1 3
2 4
-1 -1
Colors assigned to each vertex is --> 
Color of vertex 0 is 0
Color of vertex 1 is 1
Color of vertex 2 is 2
Color of vertex 3 is 0
Color of vertex 4 is 0
Total colors used-->  3

Greedy Algorithms Menu:
1. Fractional Knapsack
2. Graph Coloring
3. Exit
Enter your choice: 3
Exiting program.
________________________________________
7. Implementation of 8-Queens Problem (Backtracking)
Question: Place 8 queens on a chessboard so that no two queens attack each other.
# 8-Queens Problem using Backtracking Method (Professional Version)

def is_safe(board, row, col):
    """Check if placing a queen at board[row][col] is safe."""

    # Check column
    for r in range(row):
        if board[r] == col:
            return False

    # Check left diagonal
    for r, c in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[r] == c:
            return False

    # Check right diagonal
    for r, c in zip(range(row - 1, -1, -1), range(col + 1, 8)):
        if board[r] == c:
            return False

    return True


def solve_queens(row, board, solutions):
    """Recursive backtracking to place queens row by row."""

    # If all queens are placed
    if row == 8:
        solutions.append(board.copy())
        return

    # Try placing queen in every column
    for col in range(8):
        if is_safe(board, row, col):
            board[row] = col             # Place queen
            solve_queens(row + 1, board, solutions)
            board[row] = -1             # Backtrack


def print_board(board):
    """Print the chessboard from queen column positions."""
    for r in range(8):
        row = ["Q" if board[r] == c else "." for c in range(8)]
        print(" ".join(row))
    print()


def main():
    board = [-1] * 8              # board[row] = column of queen in that row
    solutions = []

    solve_queens(0, board, solutions)

    print(f"Total Solutions Found: {len(solutions)}\n")

    for idx, sol in enumerate(solutions, start=1):
        print(f"Solution {idx}:")
        print_board(sol)


# Run program
main()
________________________________________
8. Implementation of Traveling Salesperson Problem (Dynamic Programming)
Question: Solve TSP using dynamic programming (bitmasking approach).
n = int(input("Enter number of cities: "))
dist = []

# Input distance matrix
for i in range(n):
    value = list(map(int, input("Enter distance matrix: ").split()))
    dist.append(value)

# DP array: dp[city][mask]
dp = [[-1] * (1 << n) for _ in range(n)]

def tsp(curr_city, visited_mask):
    
    # Base case: if all cities are visited, return cost to return to start
    if visited_mask == (1 << n) - 1:
        return dist[curr_city][0]
    
    # Return memoized value
    if dp[curr_city][visited_mask] != -1:
        return dp[curr_city][visited_mask]
    
    ans = float('inf')
    
    # Try going to every unvisited city
    for next_city in range(n):
        if not (visited_mask & (1 << next_city)):     # If next_city is not visited
            new_mask = visited_mask | (1 << next_city)
            cost = dist[curr_city][next_city] + tsp(next_city, new_mask)
            ans = min(ans, cost)

    dp[curr_city][visited_mask] = ans
    return ans

# Start from city 0 with mask 1 (only city 0 visited)
start_mask = 1
result = tsp(0, start_mask)

print("\nMinimum cost is -->", result)


output
# Enter number of cities: 2
Enter distance matrix: 0 2
Enter distance matrix: 1 3

Minimum cost is --> 3

________________________________________
9. Implementation of Valid Parentheses
Question: Check if a string of parentheses is valid.
def parenthesis(s):
    st =[]
    
    for c in s :
        if c in '([{':
            st.append(c)
        elif not st :
            return False
        elif ((c ==')' and st[-1]!='(') or c == ']' and st[-1]!='[' or c == '}' and st[-1]!='{'):
            return False
        else :
            st.pop()
    return not st

s=input("Enter brackets --> ")
if parenthesis(s):
    print(s, " is valid")
else :
    print("Enter valid parenthesis")
Output:
The string '({[]})' is valid
________________________________________


