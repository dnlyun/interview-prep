C++
===

# Terminology

`Compiled`: source code is translated to machine code or bytecode before execution, resulting in an executable\
`Interpreted`: source code is translated line by line at runtime
- Property of the implementation, not the language itself (eg. C++ is compiled to .o object files -> object files are linked into an executable)

`Strongly` vs `weakly`: how strict types are enforced (eg. int + str allowed?)

`Static` vs `dynamic`: when types are checked (compile time or runtime)

`Pass by value`: function receives a copy of the variable value\
`Pass by reference`: function receives a reference to the variable

# Semantics

## Common keywords
- **Control flow**: if, else, switch, case, break, continue, return, goto, ...
- **Data types**: boolean, char, int, float, double, void, ...
- **Modifiers**: modify properties of data types
    - const: value cannot be changed after initialization
    - volatile: value can change unexpectedly, prevent certain compiler optimizations
    - signed: can be negative
    - unsigned: only non-negative
    - short/long
- **Storage classes**: specify storage duration and linkage of variables
    - auto: let compiler deduce variable type
    - extern
    - mutable: allows member of a class to be modified even if object is constant
    - register: *suggests* to compiler to store variable in CPU register for faster access
    - static
- **Functions**: specify behaviour of functions
    - inline: *suggests* to compiler to expand the function inline, reducing function call overhead
    - virtual: indicates that a function can be overridden in derived class
    - explicit: idk
- **OOP**
    - class definitions: class, struct, union, enum
    - access specifiers: public, private, protected
    - namespace management: namespace, this
    - memory management: new, delete
- **Other**: using, typedef, template, ..

## Common operators

`++`: increment\
`--`: decrement

`&&`: and operator (also an rvalue declarator)\
`||`: or operators\
`!`: not operator

`&`: address of variable\
`*`: dereference pointer\
`->`: access member of object pointed to by a pointer

`? :`: conditional expression

`,`: comma operator (`result = (expr1, expr2, ..., exprN);` evaluated left to right, returns exprN)

`::`: scope resolution operator

# Data Types

| Data Type | sizeof | Range |
| - | - | - |
| boolean | 1 byte |
| char | 1 byte | -127 to 127 or 0 to 255 |
| int | 4 bytes | -2^31 to 2^31 - 1 |
| float | 4 bytes |
| double | 8 bytes |

## Casting

`static_cast`: performs compile-time check and conversion

`dynamic_cast`

`const_cast`

`reinterpret_cast`

## Strings

### C-style strings
```cpp
#include <string.h>

int main() {
    char s1[] = "Hello"; // char str[6] = {'H', 'e', 'l', 'l', 'o', '\0'};
    char s2[] = "World";

    strcpy(s1, s2); // Copies s2 into s1 (s1 must be >= s2)
    strcat(s1, s2); // Concatenates s2 to end of s1 (s1 must be >= s1 + s2)
    strlen(s1); // Length of s1
    strcmp(s1, s2); // 0 if s1 == s2, less than 0 if s1 < s2, greater than 0 if s1 > s2

    return 0;
}
```

### String class
```cpp
#include <iostream>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = "World";

    s1 = s1 + " " + s2; // Concatenate strings

    int len = s1.size() // Length of s1

    for (char c : s1) { // Iterate over s1
        std::cout << c;
    }

    s1 = s1.substr(0, 4); // substr(size_t pos = 0, size_t len = npos)

    return 0;
}
```

# Syntax

## Preprocessor

The preprocessor processes all lines beginning with `#`

`#include` directive tells preprocessor to include the contents of that file\
Common libraries:
- iostream
- cmath
- string

`#define` directive creates symbolic constants
```cpp
#define macro replacement
```

Conditional compilation
- `#ifdef SYMBOLIC_CONSTANT` `#endif`
- `#ifndef SYMBOLIC_CONSTANT` `#endif`
- `#if 0` `#endif`

`##`: concatenate two tokens

## Variables

Variable declaration:
```cpp
int num;
int x, y, z;
```

## Structured Binding

```cpp
// Binding to data members
struct C { int x, y, z };
auto [a, b, c] = C();

// Binding a pair
std::pair<int, int> p{1, 2};
auto [p1, p2] = p;

// Iterate through map
std::unordered_map<int, int> umap;
for (const auto& [key, value] : umap) {
    // code
}
```

## Dynamic Allocation

`new`: returns a pointer to the allocated memory

`delete` or `delete[]`: deallocates memory at address pointed to (also calls destructor of the object to be deleted)

Note: when `delete` is called, pointer is not destroyed, good practice to set pointer to nullptr

## Scope

(Depends) Local variables are uninitialized when declared, global variables are zero initialized

For global variables
| Date Type | Initializer |
| - | - |
| int | 0 |
| char | '\0' |
| pointer | NULL |

```cpp
#include <iostream>

int x = 0;

int main() {
    int x = 1;
    std::cout << x << std::endl; // Prints out 1
    std::cout << ::x << std::endl; // Prints out 0
    return 0;
}
```

## Control Statements

```cpp
// for loop
for (int i = 0; i < n; ++i) {}

for (const int& num : nums) {}

for (const int& num : {0, 1, 2, 3}) {}
```

```cpp
switch(expression) {
    case a:
        // code
        break;
    case b:
        // code
        break;
    default:
        // code
}
```

## Enum

Assigns integer values starting from 0, unless explicitly assigned

### Unscoped Enums

```cpp
enum Day {
    Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday
};

int main() {
    Day day = Friday;

    int d = Sunday;

    return 0;
}
```

### Scoped Enums (enum class)

```cpp
#include <iostream>

// Can specify underlying type of an enum class
enum class Status : unsigned int {
    Ok, Error, Warning
};

enum class Color {
    Red, Green, Blue
};

int main() {
    Color c = Color::Red;
    // int value = c;   // Error: no implicit conversion

    // Explicit conversion
    int value = static_cast<int>(c);
    return 0;
}
```

# Pointers

## References vs Pointers

1. Reference cannot be NULL
2. Reference cannot be changed after initialization
3. Reference must be initialized when created

```cpp
int num = 10;

int& ref = num;

int *ptr;
ptr = &num;

// To obtain the value pointed to by a pointer, use `*`, the dereference operator
int deref = *ptr;
```

You can get the address of a pointer itself with `int **ptr2 = &ptr;`\
However, a reference does not have its own address

Passing pointer/array to function:
```cpp
// All equivalent
int add(int count, int *numbers);
int add(int count, int numbers[]);
int add(int count, int numbers[10]);

// To prevent numbers from being modified
int add(int count, const int numbers[]);
```

Returning pointer/array from function
```cpp
int* get_array() {
    int *arr = new int[5];
    return arr;
}
```

## Pointer Arithmetic

```cpp
int nums[] = {0, 1, 2, 3};

int *ptr = nums;

++ptr; // Increases by 4 (sizeof int)
ptr += 2; // Increases by 8 (2 * 4)

int *ptr2 = nums;

int num_elements_between = ptr2 - ptr1; // 3 (Pointers must be same data type)
```

## Smart Pointers

`auto_ptr`: (Deprecated after C++11, removed in C++17) automatically deletes when auto_ptr goes out of scope

`unique_ptr`:

`shared_ptr`

`weak_ptr`

# Data Structures

**C-style array**

```cpp
int num[3];
int num[3] = {0, 0, 0};
int num[] = {0, 0, 0};

// Multidimensional
int num[2][3];
int num[2][3] = {{0, 0, 0},
                 {0, 0, 0}};
int num[][3] = {{0, 0, 0},
                {0, 0, 0}};

int num[2][3][4] = {
    {
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 10, 11}
    },
    {
        {12, 13, 14, 15},
        {16, 17, 18, 19},
        {20, 21, 22, 23}
    }
};
// Flat initialization
int num[2][3][4] = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9, 10, 11,
    12, 13, 14, 15,
    16, 17, 18, 19,
    20, 21, 22, 23
};  // Equivalent
```

**std::array**

# Value Categories

`lvalue`: appears on the left-hand side of an assignment expression (can be assigned to). ie. an object that occupies some identifiable location in memory\
`rvalue`: appears on the right-hand side of an assignment expression (cannot be assigned to)\
`xvalue`: eXpiring value, refers to an object near the end of its lifetime

![](https://i.sstatic.net/GNhBF.png)

An lvalue can be implicitly converted into an rvalue
```cpp
int x = 1;
int y = x; // x implicitly converted to rvalue

int arr[3];
*(arr + 2) = 4; // arr + 2 is rvalue, but the dereference of arr + 2 is an lvalue
```

Returning an lvalue
```cpp
int global_var;

// Return an lvalue reference
int& num() {
    return global_var;
}
num() = 4;
```

Cannot take an lvalue reference of an rvalue, but...
```cpp
int& num = 10; // invalid

const& num = 10; // valid
```

```cpp
void foo(std::string& str) {} // only accepts lvalues

void foo(const std::string& str) {} // accepts lvalues and rvalues

void foo(std::string&& str) {} // only accepts rvalues
```

# Functions

## Function Declaration

```cpp
int add(int num1, int num2);
```

## Function Definition

```cpp
int add(int num1 = 0, int num2 = 0) { // Optional default parameters
    return num1 + num2;
}
```
If 1 parameter has a default value, all parameters to the right of it must also have default values

eg. `int add(int num1 = 0, int num2)` is invalid

## Function Overloading

```cpp
int add(int num1, int num2) {
    return num1 + num2;
}

int add(int num1, int num2, int num3) {
    return num1 + num2 + num3;
}

float add(float num1, float num2) {
    return num1 + num2;
}
```

## Const Functions

```cpp
class Animal {
    private:
        int age;
    public:
        int get_age() const; // Denotes that it does not modify the object for which it is called
};
```

## Variable Number of Parameters

```cpp
// Using variadic templates
template<typename... Args>
void print(Args... args) {
    ((std::cout << args), ...);
}

// C-style
#include <stdarg.h>
void print(int count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; ++i) {
        std::cout << va_arg(args, int);
    }
    va_end(args);
}
```

## Lambda Expression

```cpp
[capture](parameters) -> return_type {
    // code
}
```

**capture**: specifies which variables from the outer scope are captured
- `[x]`: captures variable x by value
- `[&x]`: captures variable x by reference
- `[x, &y]`: captures x by value, y by reference
- `[=]`: captures all variables in the surrounding scope by value
- `[&]`: captures all variables in the surrounding scope by reference
- `[=, &x]`: captures all variables by value and x by reference

### Recursive lambdas

```cpp
#include <functional> // for std::function

int main() {
    std::function<int(int)> factorial = [](int n) -> int {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    };
    return 0;
}
```

## Function as Parameter

Function that is passed to another function is called **callback**

Passing as a pointer
```cpp
// return-type (*function-name)(param1, ...)

int invoke(int x, int y, int (*f)(int, int)) {
    return f(x, y);
}

int main() {
    invoke(4, 5, &add);
    return 0;
}
```

Using function wrapper
```cpp
#include <functional>

// function<return-type(param1, ...)> function-name

int invoke(int x, int y, function<int(int, int)> f) {
    return f(x, y);
}
```

# Struct and Union

In C++, the main difference between `class` and `struct` is the default access specifier

| Feature | `class` | `struct` |
| - | - | - |
| Default member access | private | public |
| Default base class access | private (inheritance) | public (inheritance) |

Additional difference: keyword `class` can be used to declare template parameters

## Struct

```cpp
struct myStruct {
    // member
};

int main() {
    struct myStruct obj;
    return 0;
};
```

```cpp
typedef struct {
    // member
} myStruct;

int main() {
    myStruct obj;
    return 0;
}
```

## Union

All members share same memory location, ie. only one member can store a value at a time

```cpp
union Data {
    int int_value;
    float float_value;
    char char_values[10]; // Size of union is determined by largest member
};

int main() {
    Data data;

    data.int_value =  1;
    int x = data.int_value;

    data.float_value = 1.5f;
    float y = data.float_value;

    // Anonymous union
    union {
        int a;
        float b;
    };
    a = 1;
    b = 1.5f;

    return 0;
}
```

# Object Oriented Programming

## Class

Blueprint for an object

### Access modifiers
- **public**: data members and member functions are accessible from anywhere outside the class
- **private**: data members and member functions are only accessible from within the class
- **protected**: similar to private, but derived classes also have access

```cpp
class Box {
    int color; // private

    private:
        int length, width, height;

    public:
        int get_volume(); // If member function defined within a class, implicitly marked inline
};

// Define outside the class, not marked inline
// To mark as inline: inline int Box::get_volume() {}
int Box::get_volume() {
    return length * width * height;
}
```

### Constructor

`T object()` does not initialize an object; it declares a function that takes no arguments and returns T\
Before C++11, the way to value-initialize was `T object = T()`

**Direct vs list initialization**

List initialization {} prevents narrowing conversions

```cpp
double d = 1.5;

int a = d; // valid
int a(d); // valid
int a{d}; // invalid
```

**Aggregate initialization**

```cpp
struct Point {
    int x, y;
}

Point p1{1, 2}; // valid, initialize members in order
Point p2(1, 2); // invalid
```

**Member assignment:**

```cpp
class Box {
    private:
        int length, width, height;

    public:
        // Can also be defined outside the class
        Box(int l, int w, int h) {
            length = l;
            width = w;
            height = h;
        }
}
```
Each member is default-constructed first, then assigned to (inefficient)

**Member initialization**

```cpp
class Box {
    private:
        const int length, width, height;

    public:
        // Can also be defined outside the class
        Box(int l, int w, int h): length(l), width(w), height(h) {}
}

class GiftBox : public Box {
    private:
        int ribbon;

    public:
        GiftBox(int l, int w, int h, int r): Box(l, w, h), ribbon(r) {}
}
```
Value is passed into constructor (efficient)\
Must be used for const members, reference members, or members with no default constructor

**Copy constructor**

If user doesn't define a copy constructor, the compiler generates an implicit copy constructor which performs a shallow copy

```cpp
className (const className& obj) {
    // Copy logic
}
```

```cpp
class Box {
    private:
        int length, width, height;

    public:
        Box(int l, int w, int h);

        // Copy constructor
        Box(const Box& box): length(box.length), width(box.width), height(box.height) {}
}

int main() {
    Box box1(1, 2, 3);
    Box box2(box1);
    return 0;
}
```

**Assignment operator**

```cpp
class Box {
    private:
        int length, width, height;

    public:
        Box(int l, int w, int h);

        // Copy constructor
        Box& operator=(const Box& box) {
            length = box.length;
            width = box.width;
            height = box.height;
            return *this;
        }
}

int main() {
    Box box1(1, 2, 3);
    Box box2(box1);
    return 0;
}
```

Disallow copying and assignment
```cpp
class Box {
    public:
        Box& operator=(const Box& box) = delete;
        Box(const Box& box) = delete;
}
```

**Other**

See [Default arguments, overloading](#function-definition)

### Destructor

Like default constructors, automatically present in every class

```cpp
#include <iostream>

class Box {
    public:
        Box();
        ~Box(); // Can't return nor take parameters
}

Box::~Box() {
    std::cout << "Deleting box" << std::endl;
}

int main() {
    Box box;
    box.~Box(); // Explicitly calling destructor
    return 0;
} // box.~Box() automatically called here
```

### Rule of Three\Five

Rules of thumb to build exception-safe code

Rule of three
- If a class defines any of the following, then it should explicitly define all three:
    - destructor
    - copy constructor
    - copy assignment operator

Rule of five
- Extension of the rule of three for the following:
    - destructor
    - copy constructor
    - copy assignment constructor
    - move constructor
    - move assignment constructor

### static Keyword

- Independent of any object of the class, can be accessed even if no objects of the class exist
- Static member functions can only access static data members and other static member functions or functions from outside the class
- Static data members can be accessed using class name or an object

```cpp
class Box {
    public:
        static int count; // Initialized to 0
        static void reset_count();
};

// If you want to initialize outside of the class
int Box::count = 5;
void Box::reset_count() {
    count = 0;
}
```

### this Keyword

Every object has access to its own address through `this`. Implicitly passed to all non-static member functions

```cpp
class Box {
    private:
        int length, width, height;

    public:
        int get_volume();
        int compare(Box box) {
            return this->get_volume() > box.get_volume();
        }
}
```

Chaining function calls

```cpp
class Box {
    private:
        int length, width, height;

    public:
        Box(int l, int w, int h): length(l), width(w), height(h) {}
        Box& set_length(int l) {
            length = l;
            return *this;
        }
        Box& set_width(int w) {
            width = w;
            return *this;
        }
        Box& set_height(int h) {
            height = h;
            return *this;
        }
}

int main() {
    Box box(3, 4, 5);
    box.set_length(6).set_width(7).set_length(8);
    return 0;
}
```

## Principles

### Abstraction

- Hide complex implementation details and expose only essential features

### Encapsulation

- Bundle data and methods that operate on that data within a single unit
- Restrict direct access

### Inheritance

- Allows a new class to inherit attributes and behaviors from an existing class
- Promote code reusability and hierarchical organization of classes

```cpp
class derived-class : access-specifier base-class
```

| Access | public | protected | private |
| - | - | - | - |
| Same class | yes | yes | yes |
| Derived class | yes | yes | no |
| Outside class | yes | no | no |

Public inheritance
- Public members of base class become public members of derived class
- Protected members of base class become protected members of derived class

Protected inheritance
- Public and protected members of base class become protected members of derived class

Private inheritance
- Public and protected members of base class become private members of derived class

**Multiple inheritance**

```cpp
class derived-class : access baseA, access baseB...
```

Challenges:
- Ambiguity: two or more base classes have members with same name
    - Solution: scope resolution
- Diamond problem: inherits from two classes that both inherit from a common base class
    - Solution: virtual inheritance

**Multilevel inheritance**

```cpp

```

### Polymorphism

- Ability of different objects to be treated through the same interface, with behavior that adapts based on the object type

# Templates

Generic programming, allows writing functions and classes that work with different data types

```cpp
template <typename identifier> function_declaration;

template <class type> class class_name;
```

# Multithreading

**Concurrency vs parallelism**: concurrency is the ability to manage different tasks in an overlapping manner (execution may not occur simultaneously, but can overlap in time) whereas parallelism is the ability to execute tasks simultaneously (on different cores or processors)

Threads can be run on different cores, but not guaranteed and handled by the OS.

```cpp
#include <iostream>
#include <thread>

void print(int i) {
    std::cout << i << std::endl;
}

int main() {
    std::thread t(print, 0);
    t.join();
    return 0;
}
```

## Thread Synchronization

**Mutex (&lt;mutex&gt;)**
- Allows only one thread to access a shared resource at a time. Other mutex types include:
    - **shared_mutex**: has 2 locks, shared and exclusive
    - **unique_lock**: allows manual locking/unlocking, deferred locking, ownership transfer
    - **lock_guard**: a mutex wrapper with RAII mechanism. Locks only once on construction and unlocks on destruction

**Semaphore (&lt;semaphore&gt;)**
- Semaphores should be used for signalling between tasks. That is, a task that uses a semaphore should either signal or wait - not both
- Although a semaphore can be used to allow multiple threads to access a shared resource, that should still be done using mutexes
- Binary semaphore (0 or 1), counting semaphore (>= 0)

**Condition variable (&lt;condition_variable&gt;)**

**Atomic (&lt;atomic&gt;)**

**Spin Lock**

Implementation of locks: instead of putting thread to sleep while waiting, "spin" in a loop, constantly checking until the lock becomes available

# Resource Acquisition is Initialization (RAII)

# STL

## Containers

**Simple containers**: pair

**Sequence container**s: vector, list (doubly linked list), slist (singly linked list), deque (double ended queue)

**Container adapters**: stack, queue, priority queue

**Associative containers**: set, map, multiset, multimap

**Unordered associative containers**: unordered_set, unordered_map, unordered_multiset, unordered_multimap

```cpp
// Convert map to list of kv pairs
std::vector<std::pair<int, int>> vec(umap.begin(), umap.end());
```

## Algorithms

**Non-modifying**: for_each, any_of, all_of, contains, count, find, search\
**Modifying**: copy, transform, fill, remove, unique, reverse, shuffle, swap

**Sorting**: sort, stable_sort, partial_sort\
**Search**: binary_search, lower_bound, upper_bound

**Heap**: make_heap, push_heap, pop_heap, sort_heap\
**Set**: set_union, set_intersection, set_difference

**Numeric**: iota, accumulate, inner_product, adjacent_difference, partial_sum

See [cppreference](https://en.cppreference.com/w/cpp/algorithm)

## Iterators

**Input Iterators**

**Output Iterators**

**Forward Iterators**

**Bidirectional Iterators**

**Random Access Iterators**

## Functors (Function Objects)
