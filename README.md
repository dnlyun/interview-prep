# Table of Contents


# Basics

Compiled: source code is translated to machine code or bytecode before execution, resulting in an executable
Interpreted: source code is translated line by line at runtime

Strongly vs weakly: how strict types are enforced (eg. int + str allowed?)

Static vs dynamic: when types are checked (compile time or runtime)

Pass by value: function receives a copy of the variable value
Pass by reference: function receives a reference to the variable

## Built-in Types

### Truth Value Testing

An object is evaluated to `True` unless the class has a `__bool__()` method that returns `False` or `__len__()` that returns 0. Objects considered `False`:
- Constants with value `None` or `False`
- `0`, `0.0`, `0j`
- `''`, `()`, `[]`, `{}`, `set()`, `range(0)`

### Bit Manipulation

| Operation | Symbol |
| - | - |
| Bitwise AND | a & b |
| Bitwise OR | a \| b |
| Bitwise XOR | a ^ b |
| Bitwise NOT | ~a |
| Bitwise left shift | a << b |
| Bitwise right shift | a >> b |

### Float, Complex

Constructor
- `float()` accepts `'inf'` and `'nan'` (prefix optional: +/-)
- `complex(real, imag)`

Scientific notation: `3e4`, `1.4e-2`

Complex: `1 + 2j`
- `z.real` = 1.0, `z.imag` = 2.0

### Binary

Conversion
- `bin(x)`: convert int to binary string prefixed with '0b'
- `int(x, 2)`: convert binary string to int

### Strings

`chr(97)` = 'a'\
`ord('a')` = 97

`find(value, start, stop)`: first occurence of value, return -1 if value not found. rfind for last occurrence

Check if string is a number
- string.isdecimal()
- string.isdigit()
- string.isnumeric()

Check if string is alphanumeric
- string.isalnum()

`strip(characters)` (default characters = ' ')
- eg. strip(',.') to remove any leading/trailing commas and periods
- lstrip for leading and rstrip for trailing

`str.lower()`, `str.upper()`

`str.split()`, `' '.join(list)`

### F-string

`f'{a = }, {b = }, {c = }'`\
`f'{a} + {b} = {c}'` equivalent to `f'{a + b = }'`

| Format | Syntax | Result (`num = 10`) |
| - | - | - |
| 2 decimal places | `{num:.2f}` | 10.00
| hex: | `{num:#x}` | 0xa |
| binary | `{num:b}` #b to include prefix | 1010 |
| scientic notation | `{num:e}` | 1.0e+1 |
| 5 digits | `{num:05}` | 00010 |

### Print & I/O

`print(*obj, sep='', end='\n')`

| Mode | Symbol | If file doesn't exist | If file exists |
| - | - | - | - |
| Read only (default) | 'r' | Error | |
| Write only | 'w' | Create | Delete + create |
| Append | 'a' | Create | |
| Create | 'x' | | Error |

| Mode | Symbol |
| - | - |
| Text (default) | 't' |
| Binary (eg. images) | 'b' |

```python
with open('file.ext', 'r') as f:
    data = f.read()
```

## Iterator

Iterator vs Iterable: Objects like strings and lists are iterable, but not iterators

### Turn iterable into iterator

`__iter__(obj)` returns iterator object

`__iter__(obj, sentinel)`
- `obj` must be callable (check out [Objects](#objects))
- `obj` will be called until `sentinel` value is returned

```python
s = 'hello'
s = iter(s)
print(next(s))
```

### Generator

```python
def generator():
    yield 1
    yield 2
    yield 3
g = generator()
print(next(g))

def stateful_generator():
    num = yield # get input from outside
    while True:
        num = yield num + 1
sg = stateful_generator()
next(sg) # start generator
sg.send(10)
```

## Functions

### Type Hinting

```python
num: int = 0

def type_hinting(x: int, y: int | None = None) -> list[int]:
    ...
```

### Packing

```python
a, *b = 1, 2, 3
print(a, b)

name = 'First Middle Last'
first, *remaining = name.split()
print(first, remaining)

def add(*nums):
    return sum(nums)
print(add(1, 1, 1, 1, 1))
```

### Lambda functions
- Anonymous: don't require name
- Short-term use

```python
x = lambda a: a + 1
print(x(5))

y = lambda a, b: a + b
print(y(3, 4))
```

### Conditional statements

`map`

`filter`

`reduce`

### Variable Scope

A function can read a global variable but can't modify it

```python
x = 1

def foo():
    print(x)    # reads global x
    x = 2       # new local x, doesn't modify global x

def bar():
    x = x + 1   # invalid, local x referenced before assignment
```

Use `global` so local variable refers to a global variable

```python
x = 1

def foo():
    global x
    x = x + 1   # global x is modified
```
```python
def foo():
    global y    # create global variable y
    y = 2       # define locally
```

Use `nonlocal` so inner functions refer to outer function scope

```python
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 9
```

### Main function

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

### Functions are Objects

Everything in Python is an object, including functions

A callable is an object that can be called using `()`, ie. functions, objects, classes. `()` invokes `__call__()`

#### Examples

```python
def shout(s):
    print(s.upper())

def whisper(s):
    print(s.lower())
```
<details>
    <summary>Functions as objects</summary>

```python
yell = shout
yell('hi')
```
</details>
<details>
    <summary>Functions as arguments</summary>

```python
    def greet(func):
        func('hi')

    greet(whisper)
```
</details>
<details>
    <summary>Functions as return</summary>

```python
def create_adder(x):
    def adder(y):
        return x + y
    return adder
adder_2 = create_adder(2)
```
</details>

### Decorator

A function that takes another function as an argument and extends/modifies its behavior
```python
def outer(func):
    def inner(*args, **kwargs):
        print('start time:', ...)
        func(*args, **kwargs)
        print('end time:', ...)
    return inner

@outer
def foo():
    ...
```

## Class

Class variable: shared across all instances\
Instance variable: unique to each instance

Protected member: denoted by single underscore, should be accessed by class and subclass\
Private member: denoted by double underscore, should be accessed by class

`self`: Python converts obj.method(args) to ClassName.method(obj, args), so self is required to specify which instance to call on

`@staticmethod`: doesn't depend on object, callable without instantiating the class\
`classmethod`: also callable without instantiating, but follows subclass via inheritence, not super class

```python
class Person:
    species = 'homo sapien' # class variable
    def __init__(self, name='', age=1):
        self.name = name # instance variable
        self.age = age

        self._ssn = 0 # protected member

    def __str__(self):
        return f'{self.name}-{self.age}'
    
    # operator overloading
    def __add__(self, other):
        return Person(self.name + other.name, 1)
    
    @staticmethod
    def foo():
        ...

    @classmethod
    def bar(cls):
        return cls('John', 7)
```

### Inheritance

```python
class Student(Person):
    def __init__(self, name='', age=1, studentId=None):
        super().__init__(name, age) # same as Person.__init__(name, age)
        self.studentId = studentId

        self.__school = 'uw' # private member

person = Person('Optimus', 500)
student = Student('Prime', 10, 1)
baby = person + student
```

### Aditional Class Methods

`id(obj)`: unique id of object

`getattr(obj, attr: str)`: returns obj.attr, error if attr does not exist\
`getattr(obj, attr: str, default)`: returns default if attr does not exist

`setattr(obj, attr: str, value)`

### Type vs Instance

`isinstance()` supports inheritance, `type()` does not

```python
type(student) == Person     # False
isinstance(student, Person) # True
```

## Multithreading

Global interpreter lock (GIL)

# Data Structures & Algorithms

## Arrays & Tuples

`slicing [ start : stop : step ]`

`index(x[, start[, end]])`, raises error if x not found

```
# arrays: [], list()

# tuples: (), tuple()
```

### Sorting Algorithms

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
    average = best = O(nlogn), worst = O(n^2) when pivot is small or large

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

### Two Pointers & Sliding Window

```python
seq = ...

# case 1
l, r = 0, len(seq) - 1
while l <= r:
    ...
    # update pointers
    if cond1:
        l += 1
    if cond2:
        r -= 1

# case 2
l, r = 0, 0
while r < len(seq):
    ...
    # update pointers
    while cond:
        ...
        l += 1

# two pointers (process pairs of items)

# sliding window (process subarrays)
```

### Binary Search

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

#### General binary search problem

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

## Linked List

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

### Floyd's Cycle Detection

Proof:
- `x` = distance from head to cycle starting point
- `y` = distance from cycle starting point to first meeting point of both pointers
- `c` = length of cycle

1. when both pointers meet
    - slow travelled `x + y`, fast travelled `x + y + n*c`
        - fast will lap slow, therefore they will meet before slow completes 1 cycle
        - imagine both pointers are now in the loop
            - fast is 1 node behind slow: fast.next.next == slow.next -> meet
            - fast is 2 nodes behind slow: fast.next.next == slow, slow = slow.next -> now 1 node behind -> next move they meet
            - pattern continues: slow and fast will meet within c moves
2. given that the fast pointer moves twice as fast
    - `x + y + n*c = 2(x + y)`
    - `x + y = n*c`
3. reset slow and move pointers until meet again
    - when slow travels x distance, fast will be at `y + x` in the cycle
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

## Stacks

```
# use lists as stacks
stack = []
```

## Queues

```
from collections import deque

queue = deque()
```

## Heap

```
from heapq import heapify, heappop, heappush

minHeap = ...
heapify(minHeap)

maxHeap = ...
# invert values
maxHeap = [-x for x in maxHeap]
heapify(maxHeap)
```

## Set

```python
# {}, set()
```

| Method | Shortcut |
| - | - |
| set1.difference(set2) | set1 - set2 |
| set1.intersection(set2) | set1 & set2 |
| set1.union(set2) | set1 \| set2 |
| set1.issubset(set2) | set1 <= set2 |

## Dict

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

## Graph

### Search

```python
def dfs(node, visited):
    visited[node] = True

    for neighbor in node.neighbors:
        if not visited[neighbor]:
            dfs(neighbor, visited)

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

### Topological Sort

```python
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

### Shortest Path

#### Dijkstra's
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

#### Bellman-Ford
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

#### Floyd-Warshall
- Shortest path between all pairs of vertices
```python
```

### Minimum Spanning Tree

### Kruskal
- Keep adding the shortest edge to collection of components
- Time complexity
    - O(|E| * log|E|)
    - |E| <= |V|^2
    - O(|E| * log|V|^2) = O(E * log|V|)

![](https://i.sstatic.net/6RCFr.gif)
```python
def kruskal(graph):
    ...
```

#### Prim
- Add shortest edge to subgraph that doesn't create a cycle
- Time complexity O((|V| + |E|) * logV)

![](https://i.sstatic.net/KofyW.gif)
```python
def prim(graph):
    ...
```

### Connected Components

Weakly connected components - all vertices are connected by some path, ignoring direction of edges\
Strongly connected component - every pair of vertices is mutually reachable

#### Kosaraju

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

# swap incoming to outgoing edges and vice versa

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

#### Disjoint Set (union find)

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

## Trees

```
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

### Binary Search Tree

Binary tree has the following propertes:
- Node N's left child and its descendants have value lower N's value
- Node N's right child and its descendants have value higher than N's value

Inorder traversal will result in visiting the nodes by their value in increasing order

#### BST Search
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

#### BST Insert
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

#### BST Remove
- Case 1: remove leaf node
- Case 2: remove node with 1 child -> copy child node and delete child
- Case 2: remove node with 2 children -> copy inorder successor and delete inorder successor
```python
def remove(node, value):
    ...
```

#### BST Balance
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

### AVL Tree

Self-balancing BST where the height of left and right subtrees of any node cannot exceed 1

```
# search: similar to BST

# insert

# remove
```

### Red Black Tree

Provides faster insert/remove than avl, but slower lookup

```

```

### B-Tree

```

```


### Segment Tree

```

```

## Trie

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

## Dynamic Programming

### Top Down

```python
# dfs but equivalent subtrees are memoized
memo = {}
def dfs(i):
    if i in memo:
        return memo[i]
    memo[i] = dfs(i)
    return memo[i]
```

### Bottom Up

```
# tabulation

# 0/1 knapsack

# unbounded knapsack
```

# Best Practices

Naming variables
- Avoid when naming variables
    - Single letter
    - Abbreviations
- Put units in variable names

Avoid nesting
- Extraction
- Inversion

## SOLID Principles

### Single Responsibility
"A class should have only one reason to change"\
That is, each module should do one thing

### Open-Closed
"Modules should be open for extension, but closed for modification"

### Liskov Substitution
"Derived or child classes must be substitutable for their base or parent classes"\

### Interface Segregation
"Clients should not be forced to depend on interfaces it doesn't use"\
That is, instead of one large interface, have multiple smaller interfaces

### Dependency Inversion
"High-level modules should not depend on low-level modules. Both should depend on abstractions"\
Additionally, abstractions should not depend on details. Details should depend on abstractions

# Design Patterns

## Creational Design Patterns

### Factory Method

### Abstract Factory

### Singleton

### Prototype

### Builder

## Structural Design Patterns

### Adapter

### Bridge

### Composite

### Decorator

### Facade

### Flyweight

### Proxy

## Behavorial Design Patterns

### Chain of Responsibility

### Command

### Interpreter

### Mediator

### Memento

### Observer

### State

### Strategy

### Template

### Visitor

# Compilers

## Preprocessor
Before a code is compiled, it is preprocessed to perform file inclusion, macro expansion and conditional compilation. Used in languages like C, C++

For eg. the preprocessor will replace #include <stdio.h> with the textual content of stdio.h

In C/C++, files in <> will be searched in the standard compiler include paths, while "" will expand the search path to include the current source file directory

## Lexical Analysis
**Input:** source code or output of the preprocessor
- Scan left to right
    - Might need look ahead
- Group characters into lexemes (sequence of characters)
    - Each lexeme corresponds to a token
    - Ignore whitespace, comments
- Tokens
    - Identifiers (variables)
    - Keywords (int, if, return)
    - Operators (+, <, =)
    - Literals (true, 1, "Hello")
    - Seperators ((), {}, ;)
    - Comments
    - Whitespace
    
**Output:** stream of tokens

## Syntax Analysis (Parsing)
**Input:** output of lexical analysis
- Build a parse tree (hierarchial representation of code structure)
    - Detect syntax errors
        - Mismatched parentheses
        - Missing semicolons
        - Incorrect expression structure (eg. "x 3" has a missing operator)
        - etc...
- Compress parse tree into syntax tree
    
**Output:** syntax tree

## Semantic Analysis
**Input:** syntax tree
- Verify syntax tree is correct according the language grammar
    - Detect semantic errors
        - Undeclared variables
        - Invalid type conversions
        - Missing function call arguments
        - etc...
- Generate intermediate representation (IR)

**Output:** intermediate representation

## Optimization
**Input:** intermediate representation

**Output:** optimized intermediate representation

## Code Generation
**Input:** intermediate representation

**Output:** target machine code or assembly

# Operating System

## Process vs Thread

Process - a program in execution
- Each process has a separate memory space
    - Processes can communicate with each other via inter-process communication
- OS must perform full context switch when switching between processes
- Processes are managed by the process control block (PCB)
    - Created and updated by the OS for each process


Threads - smallest unit of execution in a process
- Multiple threads can exist in a single process, with shared memory space

## Concurrency

## Memory

## Cache

## Virtual vs Physical Memory

## Scheduler

## File System

## I/O

## RAID

## Virtualization & Containers

# Security

# Networks

# Database

# System Design