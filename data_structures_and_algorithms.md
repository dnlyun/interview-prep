Data Structures & Algorithms
============================

# Math

**Triangle number**: 1, 3, 6, 10, 15, 21, ...\
n-th triangle number is given by `n * (n + 1) / 2`

**Combination (nCr)**: `n! / [r! * (n - r)!]` (Note: nC2 is equal to the triangle number)

**Permutation (nPr)**: `n! / [(n - r)!]`

# Arrays & Tuples
`slicing [ start : stop : step ]`\
`index(x[, start[, end]])`, raises error if x not found
```
# arrays: [], list()

# tuples: (), tuple()
```

## Sorting Algorithms
<details>
<summary>Bubble Sort</summary>

```python
def bubble_sort(nums):
    for i in range(1, len(nums)):
        swapped = False
        for j in range(len(nums) - i):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                swapped = True
        if not swapped:
            break
    return nums
```
</details>
<details>
<summary>Selection Sort</summary>

```python
def selection_sort(nums):
    for i in range(len(nums) - 1):
        min_i = i
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[min_i]:
                min_i = j
        nums[i], nums[min_i] = nums[min_i], nums[i]
    return nums
```
</details>
<details>
<summary>Insertion Sort</summary>

```python
def insertion_sort(nums):
    for i in range(1, len(nums)):
        j = i - 1
        key = nums[i]
        while j > -1 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = key
    return nums
```
</details>
<details>
<summary>Merge Sort</summary>

```python
def merge(arr1, arr2):
    p1, p2 = 0, 0
    res = []
    while p1 < len(arr1) and p2 < len(arr2):
        if arr1[p1] < arr2[p2]:
            res.append(arr1[p1])
            p1 += 1
        else:
            res.append(arr2[p2])
            p2 += 1
    while p1 < len(arr1):
        res.append(arr1[p1])
        p1 += 1
    while p2 < len(arr2):
        res.append(arr2[p2])
        p2 += 1
    return res

def merge_sort(nums):
    if len(nums) > 1:
        mid = len(nums) // 2
        leftArr = nums[:mid]
        rightArr = nums[mid:]

        leftArr = merge_sort(leftArr)
        rightArr = merge_sort(rightArr)
        return merge(leftArr, rightArr)
    return nums
```
</details>
<details>
<summary>Quick Sort</summary>
Time complexity: average = best = O(nlogn), worst = O(n^2) when pivot is small or large

```python
def partition(nums, l, r, pivot):
    mid = nums[pivot]
    pivot = l
    while l <= r:
        if nums[l] == mid:
            l += 1
        elif nums[l] < mid:
            nums[l], nums[pivot] = nums[pivot], nums[l]
            l += 1
            pivot += 1
        else:
            nums[r], nums[l] = nums[l], nums[r]
            r -= 1
    return pivot, r

def quicksort(nums, l, r):
    if r <= l:
        return
    pivot = random.randrange(l, r)
    left, right = partition(nums, l, r, pivot) # order in-place
    quicksort(nums, l, left - 1)
    quicksort(nums, right + 1, r)
```
</details>

## Prefix Sum
Number of subarrays with sum equal to x
```python
x: int = ...
num_subarrays = 0

curr, prefix = 0, {}
for num in nums:
    curr += num
    if curr - x in prefix:
        num_subarrays += prefix[curr - x]
    prefix[curr] = prefix.get(curr, 0) + 1
return num_subarrays
```

Minimum number of removals to make array sum equal to x
```python
x: int = ...
total = sum(nums)
target = total - x
longest_subarray = 0

curr, prefix = 0, {}
for i, num in enumerate(nums):
    curr += num
    if curr - target in prefix:
        longest_subarry = max(longest_subarry, i - prefix[curr - target])
    if curr not in prefix:
        prefix[curr] = i
min_removals = len(nums) - longest_subarray
return min_removals
```

## Sliding Window
```python
l = 0
for r in range(len(arr)):
    # update pointers
    while cond:
        l += 1
```

## Two Pointers
```python
l, r = 0, len(arr) - 1
while l <= r:
    ...
    # update pointers
    if cond1:
        l += 1
    if cond2:
        r -= 1
```

## Binary Search
```python
sorted_arr = ...
target = ...

def binary_search(sorted_arr, target):
    l, r = 0, len(sorted_arr) - 1
    while l <= r:
        mid = l + (r - l) // 2 # same as (l + r) // 2, prevents overflow (useful in lang like c++)
        if sorted_arr[mid] == target: return mid
        if sorted_arr[mid] < target: l = mid + 1
        if sorted_arr[mid] > target: r = mid - 1
    return -1 # target not found
```

**General binary search problem**
Search space = [0, N]\
Objective: minimize/maximize k in search space s.t. condition(k) == True
```python
condition = ...

def minimize(arr):

    def condition(arr, idx):
        ...

    l, r = 0, len(arr) # min(search space), max(search space)
    while l < r:
        mid = l + (r - l) // 2
        if condition(arr, mid) == True:
            r = mid
        else:
            l = mid + 1
    return l # or return l - 1 depending on problem
```

# Linked List
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

'''
slow and fast pointers (variation of two pointers)
- middle of linked list
- nth node of linked list
'''

head = ...
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    ...
```

## Floyd's Cycle Detection
Proof:
- `x` = distance from head to cycle starting point
- `y` = distance from cycle starting point to first meeting point of both pointers
- `c` = length of cycle

1. When both pointers meet
    - **slow** travelled `x + y`, **fast** travelled `x + y + n*c`
        - **fast** will lap **slow**, therefore they will meet before **slow** completes 1 cycle
        - Now both pointers are in the loop
            - If **fast** is 1 node behind slow: **fast.next.next** == slow.next -> meet
            - If **fast** is 2 nodes behind slow: **fast.next.next** == **slow**, **slow** = **slow.next** -> now 1 node behind -> next move they meet
            - Pattern continues: **slow** and **fast** will meet within c moves
2. Given that the **fast** pointer moves twice as **slow**
    - `x + y + n*c = 2(x + y)`
    - `x + y = n*c`
3. Reset **slow** and move pointers until meet again
    - When **slow** travels x distance, **fast** will be at `y + x` in the cycle
    - `y + x = n*c`, n complete laps -> finish at the cycle starting point
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        break
slow = head
while slow != fast:
    slow = slow.next
    fast = fast.next
# slow is at cycle starting point
```

# Stacks
```python
# use lists as stacks
stack = []
```

# Queues
```python
from collections import deque

queue = deque()
```

# Heap
```python
from heapq import heapify, heappop, heappush

minHeap = ...
heapify(minHeap)

maxHeap = ...
# invert values
maxHeap = [-x for x in maxHeap]
heapify(maxHeap)
```

# Set
```python
# {}, set()
```

| Method | Shortcut |
| - | - |
| set1.difference(set2) | set1 - set2 |
| set1.intersection(set2) | set1 & set2 |
| set1.union(set2) | set1 \| set2 |
| set1.issubset(set2) | set1 <= set2 |

# Dict
```python
# {}, dict()
```

| Method | Shortcut |
| - | - |
| dict1.update(dict2) | dict1 \| dict2 |

Returns view object, not list or iterator
- dict1.keys()
- dict1.values()
- dict1.items()

# Graph

## Depth First Search
```python
def dfs(node, visited):
    visited[node] = True

    for neighbor in node.neighbors:
        if not visited[neighbor]:
            dfs(neighbor, visited)
```
Cycle detection using dfs
```python
# cycle detection using dfs
def isCyclic(node, visited, curr_path):
    if visited[node]:
        return False

    if curr_path[node]:
        return True

    curr_path[node] = True

    for neighbor in node.neighbors:
        if isCyclic(neighbor, visited, curr_path):
            return True

    visited[node] = True
    curr_path[node] = False
    return False
```

## Breadth First Search
```python
def bfs(node):
    queue = deque([node])
    visited = ...

    while queue:
        n = queue.popleft()
        visited[n] = True
        for neighbor in n.neighbors:
            if not visited[neighbor]:
                queue.append(neighbor)
```

## Topological Sort
```python
# Kahn's algo
def topological_sort(graph):
    in_deg = {node: 0 for node in graph}

    for node in graph:
        for neighbor in node.neighbors:
            in_deg[neighbor] += 1

    in_deg_0 = deque([node for node in in_deg if in_deg[node] == 0])

    topo_order = []
    while in_deg_0:
        node = in_deg_0.popleft()
        topo_order.append(node)

        for neighbor in node.neighbors:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                topo_order.append(neighbor)

    if len(topo_order) != n:
        # contains cycle, cannot be topo sorted
        return
    return topo_order
```

## Connected Components
Weakly connected components - all vertices are connected by some path, ignoring direction of edges\
Strongly connected component - every pair of vertices is mutually reachable

### Kosaraju's
1. Run dfs, push node onto stack once it's finished
2. Reverse the direction of all edges in the graph
3. Run dfs in order of the nodes on stack, giving us one SCC
```python
nodes = ...

visited = ...
stack = []

def dfs(node, visited):
    visited[node] = True
    for neighbor in node.neighbors:
        if not visited[neighbor]:
            dfs(neighbor, visited)
    stack.append(node)

for node in nodes:
    if not visited[node]:
        dfs(node, visited)

# create reversed graph
# swap incoming and outgoing edges

scc = 0
scc_map = ...

def dfs_reversed(node, visited, scc_map, scc):
    visited[node] = False
    scc_map[node] = scc
    for neighbor in node.nieghbors:
        if visited[neighbor]:
            dfs_reversed(neighbor, visited)

while stack:
    node = stack.pop()
    if visited[node]:
        dfs_reversed(node, visited, scc_map, scc)
        scc += 1
```

### Disjoint Set (Union Find)
1. Mark each node's parent as itself (each node is in its own set)
2. If two nodes are merged, update one of the nodes' parent to the other
```python
n = ... # num nodes
parent = list(range(n))

# optimization: union by rank
rank = [0] * n

def find(x):
    if parent[x] == x:
        return x

    # path compression
    p = find(parent[x])

    parent[x] = p
    return p

def union(x, y):
    x_parent = find(x)
    y_parent = find(y)

    if x_parent == y_parent:
        return

    if rank[x_parent] < rank[y_parent]:
        parent[x_parent] = y_parent
    elif rank[y_parent] < rank[x_parent]:
        parent[y_parent] = x_parent
    else:
        parent[x_parent] = y_parent
        rank[y_parent] += 1
```

## Shortest Path

### Dijkstra's
- Shortest path from source vertex
- Time copmlexity
    - O((|V| + |E|) * logV) with priority queue
    - O(|V|^2) with array
```Python
def dijkstra(graph, src):
    dist = [float('inf')] * n

    visited = ...
    queue = [(0, src)]
    while queue:
        d, node = heappop(queue)

        if visited[node]:
            continue

        visited[node] = True
        dist[node] = d

        for neighbor, weight in node.neighbors:
            if not visited[neighbor] and d + weight < dist[neighbor]:
                heappush(queue, (d + weight, neighbor))
    return dist
```

### Bellman-Ford
- Shortest path from source vertex
- Can handle negative weights and detect negative cycles
- Time complexity O(|V| * |E|)
```python
def bellman_ford(graph, src):
    dist = [float('inf')] * n
    dist[src] = 0

    for i in range(n - 1):
        for node in graph:
            for neighbor, weight in node.neighbors:
                if dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight

    # check negative cycle
    for node in graph:
        for neighbor, weight in node.neighbors:
            if dist[node] + weight < dist[neighbor]:
                # shortest path can be improved -> contains negative cycle
                ...

    return dist
```

### Floyd-Warshall
- Shortest path between all pairs of vertices
```python

```

## Minimum Spanning Tree

### Kruskal's
- Keep adding the shortest edge to collection of components
- Time complexity
    - O(|E| * log|E|)
    - |E| <= |V|^2
    - O(|E| * log|V|^2) = O(E * log|V|)

![](https://i.sstatic.net/6RCFr.gif)
```python
def kruskal(graph):
    def find(x):
        ...
    def union(x, y):
        ...

    n = ... # num nodes
    parent = list(range(n))
    rank = [0] * n

    mst_cost = 0
    mst = set()
```

### Prim's
- Add shortest edge to subgraph that doesn't create a cycle
- Time complexity O((|V| + |E|) * logV)

![](https://i.sstatic.net/KofyW.gif)
```python
def prim(graph):
    visited = [False] * n

    mst_cost = 0
    mst = set()

    pq = [(0, 0, -1)] # cost, node 0, parent node
    while pq:
        cost, node, parent = heappop(pq)

        if visited[node]:
            continue
        visited[node] = True

        mst_cost += cost
        if parent != -1:
            mst.add((parent, node))

        for neighbor, weight in node.neighbors:
            if not visited[neighbor]:
                heappush(pq, (weight, neighbor, node))
    return mst, mst_cost if sum(visited) == n else set(), -1
```

# Trees
```python
# binary tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

'''
n-ary tree

class TreeNode:
    def __init__(self, val=0, children: List[TreeNode]=None):
        self.val = val
        self.children = children
'''
```
```python
# preorder traversal
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# inorder traversal
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# postorder traversal
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

## Binary Search Tree
Binary tree has the following propertes:
- Node N's left child and its descendants have value lower N's value
- Node N's right child and its descendants have value higher than N's value

Inorder traversal will result in visiting the nodes by their value in increasing order

### BST Search
```python
def search(node, value):
    if not node:
        return None
    elif node.val == value:
        return node
    elif node.val < value:
        return search(node.right, value)
    else:
        return search(node.left, value)
```
Time complexity: O(h)

### BST Insert
```python
def insert(node, value):
    if not node:
        return TreeNode(value)
    elif node.val < value:
        node.right = insert(node.right, value)
    elif node.val > value:
        node.left = insert(node.left, value)
    return node
```
Time complexity: O(h)

### BST Remove
- Case 1: remove leaf node
- Case 2: remove node with 1 child -> copy child node and delete child
- Case 2: remove node with 2 children -> copy inorder successor and delete inorder successor
```python
def remove(node, value):
    ...
```

### BST Balance
- Do inorder traversal, make root node the middle item, recurse for left and right
- Balancing BST will reduce height, optimizing search/insert/remove
```python
def balance(node):
    def create_subtree(arr, l, r):
        if r < l:
            return
        mid = l + (r - l) // 2
        root = TreeNode(arr[mid],
                        create_subtree(arr, l, mid - 1),
                        create_subtree(arr, mid + 1, r))
        return root

    arr = inorder(node)
    return create_subtree(arr, 0, len(arr) - 1)
```

## AVL Tree
Self-balancing BST where the height of left and right subtrees of any node cannot exceed 1
```
# search: similar to BST

# insert

# remove
```

## Red Black Tree
Provides faster insert/remove than avl, but slower lookup
```

```

## B-Tree
```

```


## Segment Tree
```

```

# Trie
```python
class Trie:
    def __init__(self):
        self.trie = {}

    def insert(self, word):
        curr = self.trie
        for letter in word:
            if letter not in curr:
                curr[letter] = {}
            curr = curr[letter]
        curr['end'] = True

    def search(self, word):
        curr = self.trie
        for letter in word:
            if letter not in curr:
                return False
            curr = curr[letter]
        return curr.get('end')
```

# Dynamic Programming

## Backtrack
```python
def backtrack(state):
    if is_solution(state):
        output(solution)
        return

    for choice in valid_choices(state):
        make_choice(state, choice)
        backtrack(state)
        undo_choice(state, choice)
```

## Top Down
```python
# dfs but equivalent subtrees are memoized
memo = {}
def fib(n):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
```

## Bottom Up
Fibonnaci
```python
def fib(n):
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a
```
0/1 knapsack
```python
# n items, c capacity
def knapsack(weight, value, c, n):
    dp = [0] * (c + 1)
    for i in range(n):
        for j in range(c, weight[i] - 1, -1):
            dp[j] = max(dp[j], value[i] + dp[j - weight[i]])
    return dp[-1]
```
Note:
- Iterate inner loop backwards to avoid double counting
- If counting same item mulitple times is allowed (unbounded knapsack), iterate inner loop forwards

Unbounded knapsack
```python
# n items, c capacity
def knapsack_combinations(weight, c, n):
    dp = [0] * (c + 1)
    dp[0] = 1
    for i in range(n):
        for j in range(weight[i], c + 1):
            dp[j] += dp[j - weight[i]]
    return dp[-1]
```
Note:
- This gives us the total combinations to reach capacity j
    - For each value, only combinations that include that value after previous ones are counted
- If we iterate over weight first then values, we get total permutations

Longest increasing subsequence
```python
def lis(nums):
    dp = [1] * n
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], 1 + dp[j])
    return dp[-1]

'''
patience sorting
- if num is greater than value of top item in pile, create new pile
- else place it in leftmost pile where it can be placed on top
'''
def lis(nums):
    piles = [nums[0]]
    for num in nums[1:]:
        if num > piles[-1]:
            piles.append(num)
        else:
            l, r = 0, len(piles) - 1
            while l <= r:
                mid = (l + r) // 2
                if piles[mid] == num:
                    break
                elif piles[mid] < num:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                piles[l] = num
    return len(piles)
```

Longest common subsequence