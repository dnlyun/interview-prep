# Operating System

## Stack vs Heap
Stack: region of memory that stores temporary variables
- LIFO structure
- Fast access
- Managed automatically
- Fixed size

Heap: region of memory used for dynamic memory allocation
- user/program manually allocates/frees memory (or language's runtime/garbage collector does it for you)
- slower access
- Managed explicitly (not by OS)
- Larger than stack, can grow

## Process vs Thread
Process: a program in execution
- Each process has a separate memory space
    - Processes can communicate with each other via inter-process communication
- OS must perform full context switch when switching between processes
- Processes are managed by the process control block (PCB)
    - Created and updated by the OS for each process


Threads: smallest unit of execution in a process
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

# Security

# Networks

# Database
