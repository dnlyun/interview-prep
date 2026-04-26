# C++

# Terminology

`Compiled`: source code is translated to machine code before execution, resulting in an executable\
`Interpreted`: source code is translated line by line at runtime
- Property of the implementation, not the language itself (e.g. C++ is compiled to `.o` object files → linked into an executable)

`Strongly` vs `weakly typed`: how strictly types are enforced (e.g. is `int + str` allowed?)

`Static` vs `dynamic typed`: when types are checked (compile time vs runtime)

`Pass by value`: function receives a copy of the argument\
`Pass by reference`: function receives a reference to the original variable

# Semantics

## Common Keywords

- **Control flow**: `if`, `else`, `switch`, `case`, `break`, `continue`, `return`, `goto`
- **Data types**: `bool`, `char`, `int`, `float`, `double`, `void`
- **Modifiers**: modify properties of data types
    - `const`: value cannot be changed after initialization
    - `volatile`: value may change unexpectedly; prevents certain compiler optimizations
    - `signed`: can be negative
    - `unsigned`: non-negative only
    - `short` / `long`
- **Storage classes**: specify storage duration and linkage of variables
    - `auto`: let the compiler deduce the variable type
    - `extern`: declares a variable or function defined in another translation unit
    - `mutable`: allows a class member to be modified even if the object is `const`
    - `register`: *suggests* to the compiler to store the variable in a CPU register
    - `static`: persists for the program's lifetime; limits linkage to the current file at file scope
- **Function specifiers**
    - `inline`: *suggests* to the compiler to expand the function inline
    - `virtual`: indicates a function can be overridden in a derived class
    - `explicit`: prevents the compiler from using a constructor for implicit type conversions
    - `override`: explicitly marks a virtual function as overriding a base class function (C++11)
    - `final`: prevents further overriding or inheritance (C++11)
- **OOP**
    - Class definitions: `class`, `struct`, `union`, `enum`
    - Access specifiers: `public`, `private`, `protected`
    - Namespace management: `namespace`, `this`
    - Memory management: `new`, `delete`
- **Other**: `using`, `typedef`, `template`, `static_assert`

## Common Operators

`++`: increment\
`--`: decrement

`&&`: logical AND (also used as rvalue reference declarator in type context)\
`||`: logical OR\
`!`: logical NOT

`&`: address-of operator\
`*`: dereference operator\
`->`: member access through a pointer

`? :`: ternary conditional expression

`,`: comma operator — `result = (expr1, expr2, ..., exprN)` evaluates left-to-right, yields `exprN`
- `for (int i = 0, j = 10; i < j; ++i, --j) { ... }`

`::`: scope resolution operator

`sizeof`: returns the size of a type or object in bytes

# Data Types

| Data Type  | Size     | Range / Notes                      |
| ---------- | -------- | ---------------------------------- |
| `bool`     | 1 byte   | `true` (1) or `false` (0)          |
| `char`     | 1 byte   | -128 to 127 or 0 to 255            |
| `int`      | 4 bytes  | -2³¹ to 2³¹ − 1                   |
| `float`    | 4 bytes  | ~6–7 significant decimal digits    |
| `double`   | 8 bytes  | ~15–16 significant decimal digits  |
| `long long`| 8 bytes  | -2⁶³ to 2⁶³ − 1                   |

Use `<cstdint>` for fixed-width types: `int8_t`, `int32_t`, `uint64_t`, etc.

## Endianness

**Byte order** — the order in which bytes of a multi-byte value are stored in memory.

- **Little-endian**: least significant byte stored at the lowest address (x86, ARM default)
- **Big-endian**: most significant byte stored at the lowest address (network byte order, some RISC)

Example: storing `0x01020304` at address `0x00`:

| Address | Little-endian | Big-endian |
| ------- | ------------- | ---------- |
| `0x00`  | `04`          | `01`       |
| `0x01`  | `03`          | `02`       |
| `0x02`  | `02`          | `03`       |
| `0x03`  | `01`          | `04`       |

### Detecting endianness at runtime

```cpp
#include <cstdint>

bool is_little_endian() {
    uint32_t val = 1;  // 0x00000001
    uint8_t* byte = reinterpret_cast<uint8_t*>(&val);
    return *byte == 1;
}
```

**Why this works**: The integer `1` is `0x00000001` — the least significant byte is `0x01` and the rest are `0x00`. By casting to a `uint8_t*`, we read just the first byte at the lowest address. On a little-endian system the least significant byte (`0x01`) is stored first, so `*byte == 1`. On a big-endian system the most significant byte (`0x00`) is stored first, so `*byte == 0`.

### Using a char pointer

```cpp
bool is_little_endian() {
    int val = 1;
    char* byte = (char*)&val;
    return *byte == 1;
}
```

Same idea — `char*` is allowed to alias any type in C/C++, so casting `int*` to `char*` and reading the first byte is well-defined (unlike `reinterpret_cast` to most other types, which can be undefined behavior).

## Casting

`static_cast`: compile-time checked conversion between compatible types
```cpp
double d = 3.14;
int i = static_cast<int>(d); // 3
```

`dynamic_cast`: safe downcasting in an inheritance hierarchy; requires at least one virtual function in the base class
```cpp
Base* b = new Derived();
Derived* d = dynamic_cast<Derived*>(b); // returns nullptr if cast fails (pointer version)
                                        // throws std::bad_cast if cast fails (reference version)
```

`const_cast`: adds or removes `const` from a pointer or reference
```cpp
const int x = 5;
int* p = const_cast<int*>(&x); // removes const; writing through p is undefined behavior
```

`reinterpret_cast`: reinterprets the bit pattern of a value as a different type; unsafe
```cpp
int x = 42;
char* p = reinterpret_cast<char*>(&x);
```

## Strings

### C-style Strings
```cpp
#include <cstring>

int main() {
    char s1[10] = "Hello";  // {'H', 'e', 'l', 'l', 'o', '\0', '\0', '\0', '\0', '\0'}
    char s2[] = "World";    // {'W', 'o', 'r', 'l', 'd', '\0'}

    strcpy(s1, s2);   // copies s2 into s1 (s1 must be large enough)
    strcat(s1, s2);   // appends s2 to s1
    strlen(s1);       // length (excluding null terminator)
    strcmp(s1, s2);   // 0 if equal, <0 if s1 < s2, >0 if s1 > s2

    return 0;
}
```

### `std::string`
```cpp
#include <iostream>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = "World";

    s1 = s1 + " " + s2;         // concatenation
    int len = s1.size();         // length
    s1.find("World");            // returns index or std::string::npos
    s1 = s1.substr(0, 5);        // substr(pos, len)

    for (char c : s1) {
        std::cout << c;
    }

    return 0;
}
```

# Syntax

## Preprocessor

The preprocessor processes all lines beginning with `#` before compilation.

`#include`: inserts the contents of a header file\
Common headers: `<iostream>`, `<cmath>`, `<string>`, `<vector>`, `<algorithm>`

`#define`: creates a macro or symbolic constant
```cpp
#define PI 3.14159
#define SQUARE(x) ((x) * (x))
```

Conditional compilation:
```cpp
#ifdef SYMBOLIC_CONSTANT
// ...
#endif

#ifndef SYMBOLIC_CONSTANT
// ...
#endif

#if 0
// Disabled code
#endif
```

`##`: token concatenation operator
```cpp
#define CONCAT(a, b) a##b
int CONCAT(my, Var) = 5; // expands to: int myVar = 5;
```

## Variables

```cpp
int num;
int x = 5, y = 10;
auto z = 3.14; // type deduced as double
```

## Structured Bindings (C++17)

```cpp
// Bind to struct members
struct Point { int x, y, z; };
auto [a, b, c] = Point();

// Bind a pair
std::pair<int, int> p{1, 2};
auto [p1, p2] = p;

// Iterate over a map
std::unordered_map<int, int> umap;
for (const auto& [key, value] : umap) {
    // ...
}
```

## Scope

Local variables are uninitialized when declared; global variables are zero-initialized.

| Data Type | Default value |
| --------- | ------------- |
| `int`     | `0`           |
| `char`    | `'\0'`        |
| pointer   | `nullptr`     |

```cpp
#include <iostream>

int x = 0;

int main() {
    int x = 1;
    std::cout << x << std::endl;   // 1 (local)
    std::cout << ::x << std::endl; // 0 (global)
    return 0;
}
```

## Control Statements

```cpp
// Range-based for loop
for (const int& num : nums) {}

for (const int& num : {0, 1, 2, 3}) {}

// Index-based for loop
for (int i = 0; i < n; ++i) {}
```

```cpp
switch (expression) {
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

Assigns integer values starting from 0 unless explicitly set.

### Unscoped Enums

```cpp
enum Day {
    Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday
};

int main() {
    Day day = Friday;
    int d = Sunday; // implicit conversion to int
    return 0;
}
```

### Scoped Enums (`enum class`)

No implicit conversion to int; must use scope qualifier.

```cpp
#include <iostream>

enum class Status : unsigned int {
    Ok, Error, Warning
};

enum class Color {
    Red, Green, Blue
};

int main() {
    Color c = Color::Red;
    // int value = c;                   // Error: no implicit conversion
    int value = static_cast<int>(c);    // explicit conversion
    return 0;
}
```

## Namespaces

Group related declarations to avoid name collisions.

```cpp
namespace math {
    int add(int a, int b) { return a + b; }
    const double PI = 3.14159;
}

int main() {
    math::add(1, 2);

    using namespace math; // imports all names into current scope (can cause collisions)
    add(1, 2);

    using math::add; // import only add
    return 0;
}
```

Namespaces can be nested and reopened:
```cpp
namespace outer {
    namespace inner {
        void foo() {}
    }
}
outer::inner::foo();

namespace outer::inner { // C++17 shorthand
    void bar() {}
}
```

## Type Deduction

### `auto`

Lets the compiler deduce the type from the initializer.
```cpp
auto i = 42;                  // int
auto d = 3.14;                // double
auto s = std::string("hello"); // std::string
auto& ref = i;                // int&
```

### `decltype`

Deduces the type of an expression without evaluating it.
```cpp
int x = 5;
decltype(x) y = 10;       // int
decltype(x + 3.0) z = 0;  // double
```

Useful in templates:
```cpp
template<typename A, typename B>
auto add(A a, B b) -> decltype(a + b) {
    return a + b;
}
```

## `constexpr`

Evaluated at compile time. More powerful than `#define` or `const`.

```cpp
constexpr int square(int x) { return x * x; }

constexpr int arr_size = 10;
int arr[arr_size]; // valid: size known at compile time

constexpr int s = square(5); // 25, computed at compile time
```

`consteval` (C++20): forces evaluation at compile time only.

# Pointers

## References vs Pointers

1. A reference cannot be `nullptr`
2. A reference cannot be rebound after initialization
3. A reference must be initialized when declared

```cpp
int num = 10;

int& ref = num;  // reference
int* ptr = &num; // pointer

int deref = *ptr; // dereference
```

You can get the address of a pointer with `int** ptr2 = &ptr`; a reference has no address of its own.

Passing arrays to functions:
```cpp
// All equivalent
int sum(int count, int* numbers);
int sum(int count, int numbers[]);
int sum(int count, int numbers[10]);

// Prevent modification
int sum(int count, const int numbers[]);
```

Returning a pointer from a function:
```cpp
int* get_array() {
    int* arr = new int[5];
    return arr; // caller must delete[]
}
```

## Pointer Arithmetic

```cpp
int nums[] = {0, 1, 2, 3};

int* ptr = nums;

++ptr;    // advances by sizeof(int) = 4 bytes
ptr += 2; // advances by 8 bytes

int* ptr2 = nums;

int diff = ptr - ptr2; // 3 (number of elements between them)
```

## Dynamic Allocation

`new`: allocates memory on the heap and returns a pointer to it\
`delete` / `delete[]`: deallocates heap memory and calls the object's destructor

```cpp
int* p = new int(5);
delete p;
p = nullptr; // good practice: avoid dangling pointer

int* arr = new int[10];
delete[] arr;
```

Note: prefer smart pointers over raw `new`/`delete` to avoid memory leaks.

## Smart Pointers (`<memory>`)

Prefer smart pointers over raw `new`/`delete`.

### `unique_ptr`

Sole ownership of the resource. Non-copyable; moveable.
```cpp
#include <memory>

auto p = std::make_unique<int>(42);
// auto p2 = p;          // Error: cannot copy
auto p2 = std::move(p);  // ownership transferred; p is now nullptr
```

### `shared_ptr`

Shared ownership via reference counting. Deletes when the count reaches 0.
```cpp
auto p1 = std::make_shared<int>(42);
auto p2 = p1; // both own the object; ref count = 2
// deleted when both p1 and p2 go out of scope
```

### `weak_ptr`

Non-owning reference to a `shared_ptr`-managed object. Used to break circular references.
```cpp
std::shared_ptr<int> sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;

if (auto locked = wp.lock()) { // returns shared_ptr if still alive
    std::cout << *locked;
}
```

# Data Structures

## C-style Array

```cpp
int nums[3];
int nums[3] = {0, 0, 0};
int nums[] = {0, 0, 0};

// Multidimensional
int grid[2][3] = {{0, 1, 2}, {3, 4, 5}};
int grid[][3]  = {{0, 1, 2}, {3, 4, 5}}; // first dimension can be omitted

int cube[2][3][4] = {
    {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}},
    {{12, 13, 14, 15}, {16, 17, 18, 19}, {20, 21, 22, 23}}
};
```

## `std::array` (`<array>`)

Fixed-size, stack-allocated. Size known at compile time. Safer than C-style arrays.
```cpp
#include <array>

std::array<int, 5> arr = {1, 2, 3, 4, 5};
arr.size();     // 5
arr.at(2);      // bounds-checked access
arr[2];         // unchecked access
arr.front();    // first element
arr.back();     // last element
arr.fill(0);    // set all elements to 0
```

## `std::vector` (`<vector>`)

Dynamic array; grows automatically.
```cpp
#include <vector>

std::vector<int> v = {1, 2, 3};
v.push_back(4);
v.pop_back();
v.size();
v.empty();
v.reserve(100);               // pre-allocate capacity
v.resize(10);                 // resize (fills new elements with 0)
v.insert(v.begin() + 1, 99); // insert at position
v.erase(v.begin());           // erase first element
```

# Functions

## Declaration and Definition

```cpp
int add(int a, int b); // forward declaration (declare before defining)

int add(int a = 0, int b = 0) { // definition with default parameters
    return a + b;
}
```

Parameters with defaults must be rightmost:
```cpp
int add(int a = 0, int b); // invalid
int add(int a, int b = 0); // valid
```

## Function Overloading

```cpp
int add(int a, int b)        { return a + b; }
int add(int a, int b, int c) { return a + b + c; }
float add(float a, float b)  { return a + b; }
```

## Const Member Functions

```cpp
class Animal {
    int age;
public:
    int get_age() const; // promises not to modify the object
};
```

## Variadic Functions

```cpp
// Variadic templates (preferred, type-safe)
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << ' '), ...); // fold expression (C++17)
}

// C-style variadic (legacy)
#include <cstdarg>
void print(int count, ...) {
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; ++i) {
        std::cout << va_arg(args, int);
    }
    va_end(args);
}
```

## Lambda Expressions

```cpp
[capture](parameters) -> return_type {
    // body
}
```

**Capture list:**
- `[x]`: capture `x` by value
- `[&x]`: capture `x` by reference
- `[x, &y]`: capture `x` by value, `y` by reference
- `[=]`: capture all reachable variables by value
- `[&]`: capture all reachable variables by reference
- `[=, &x]`: capture all by value, `x` by reference

```cpp
auto square = [](int x) { return x * x; };
auto add = [](int a, int b) -> int { return a + b; };
```

### Recursive Lambdas

```cpp
#include <functional>

std::function<int(int)> factorial = [&factorial](int n) -> int {
    return n <= 1 ? 1 : n * factorial(n - 1);
};
```

## Callbacks and `std::function`

```cpp
// Function pointer
int invoke(int x, int y, int (*f)(int, int)) {
    return f(x, y);
}

// std::function wrapper (accepts functions, lambdas, and functors)
#include <functional>

int invoke(int x, int y, std::function<int(int, int)> f) {
    return f(x, y);
}
```

# Value Categories

`lvalue`: expression identifying a persistent object in memory (can appear on the left-hand side of an assignment)\
`rvalue`: temporary value with no persistent memory location\
`xvalue`: "eXpiring value" — an rvalue whose resources can be moved from

![](https://i.sstatic.net/GNhBF.png)

An lvalue can be implicitly converted to an rvalue:
```cpp
int x = 1;
int y = x; // x used as rvalue here

int arr[3];
*(arr + 2) = 4; // (arr + 2) is an rvalue, but *(arr + 2) is an lvalue
```

Returning an lvalue reference:
```cpp
int global_var;

int& get() {
    return global_var;
}
get() = 4; // assigns 4 to global_var
```

Cannot bind an lvalue reference to an rvalue, but a `const` lvalue reference can:
```cpp
int& a = 10;        // Error
const int& b = 10;  // OK: const lvalue reference extends the lifetime of the temporary
```

Function overloads by value category:
```cpp
void foo(std::string& str)       {} // lvalues only
void foo(const std::string& str) {} // lvalues and rvalues
void foo(std::string&& str)      {} // rvalues only
```

# Move Semantics

Move semantics (C++11) allow transferring resources from a temporary object rather than copying, avoiding unnecessary allocations.

## `std::move`

Casts an lvalue to an rvalue reference, enabling a move instead of a copy.
```cpp
#include <utility>

std::string a = "hello";
std::string b = std::move(a); // a's content is moved into b; a is now in a valid but unspecified state
```

## Move Constructor and Move Assignment Operator

```cpp
class Buffer {
    int* data;
    size_t size;
public:
    // Move constructor
    Buffer(Buffer&& other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment operator
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};
```

## Perfect Forwarding

`std::forward` preserves the value category of a forwarded argument. Used with forwarding references (`T&&`) in templates.
```cpp
#include <utility>

template<typename T>
void wrapper(T&& arg) {
    target(std::forward<T>(arg)); // forwards as lvalue if T is lvalue ref, rvalue otherwise
}
```

# Struct and Union

In C++, the main difference between `class` and `struct` is the default access specifier:

| Feature                   | `class`             | `struct`            |
| ------------------------- | ------------------- | ------------------- |
| Default member access     | `private`           | `public`            |
| Default base class access | `private`           | `public`            |

`class` can also declare template type parameters; `struct` cannot.

## Struct

In C, you must use the `struct` keyword when declaring variables: `struct Point p;`. A common workaround is `typedef`:
```c
typedef struct Point {
    int x, y;
} Point;

Point p; // no struct prefix needed
```

In C++, the `struct` tag is automatically usable as a type name — no `typedef` needed. However, if a function shares the same name as a struct, the function hides the type and you must disambiguate with the `struct` keyword:
```cpp
struct Point { int x, y; };
void Point() {} // function hides the struct name

int main() {
    struct Point p = {1, 2}; // must use 'struct' prefix to refer to the type
    return 0;
}
```

## Union

All members share the same memory location; only one member holds a value at a time.

```cpp
union Data {
    int   i;
    float f;
    char  s[10]; // union size = size of largest member
};

int main() {
    Data d;
    d.i = 1;
    d.f = 1.5f; // overwrites i

    // Anonymous union
    union {
        int a;
        float b;
    };
    a = 1;
    return 0;
}
```

# Object-Oriented Programming

## Class

Blueprint for creating objects.

### Access Specifiers

| Specifier   | Same class | Derived class | Outside class |
| ----------- | ---------- | ------------- | ------------- |
| `public`    | yes        | yes           | yes           |
| `protected` | yes        | yes           | no            |
| `private`   | yes        | no            | no            |

```cpp
class Box {
    int color; // private by default

    private:
        int length, width, height;

    public:
        int get_volume(); // implicitly inline if defined in class declaration
};

// Define outside the class, not marked inline
// To mark as inline: inline int Box::get_volume() {}
int Box::get_volume() {
    return length * width * height;
}
```

### Constructors

`T object()` does not create an object; it declares a function (most vexing parse)\
Use `T object{}` or `T object = T()` for value initialization.

**List initialization `{}`** prevents narrowing conversions:
```cpp
double d = 1.5;
int a = d;  // OK
int b(d);   // OK
int c{d};   // Error: narrowing conversion
```

**Aggregate initialization** (structs and classes with no user-provided constructors):
```cpp
struct Point { int x, y; };
Point p{1, 2}; // OK: initializes members in order
```

**Member initialization list** (preferred):
```cpp
class Box {
    const int length, width, height;
public:
    Box(int l, int w, int h) : length(l), width(w), height(h) {}
};
```
Required for `const` members, reference members, and members with no default constructor. More efficient than assignment in the constructor body.

**Copy constructor**:
```cpp
class Box {
    int length, width, height;
public:
    Box(int l, int w, int h);
    Box(const Box& other) : length(other.length), width(other.width), height(other.height) {}
};
```
If not defined, the compiler generates an implicit shallow copy.

**Assignment operator**:
```cpp
Box& operator=(const Box& other) {
    if (this != &other) {
        length = other.length;
        width  = other.width;
        height = other.height;
    }
    return *this;
}
```

**Deleting copy and assignment**:
```cpp
class Box {
public:
    // explicitly disable specific default functions
    Box(const Box&) = delete;
    Box& operator=(const Box&) = delete;
};
```

### Destructor

Called automatically when an object leaves scope or `delete` is used. Cannot return a value or take parameters.

```cpp
class Box {
public:
    ~Box() { std::cout << "Box destroyed\n"; }
};

int main() {
    Box b;
    b.~Box(); // explicit destructor call (rare)
    return 0;
}
```

### Rule of Three / Five / Zero

**Rule of Three**: if a class defines any of the following, it should define all three:
- Destructor
- Copy constructor
- Copy assignment operator

**Rule of Five** (C++11): extends the Rule of Three with:
- Move constructor
- Move assignment operator

**Rule of Zero**: design classes so they need none of the above (use smart pointers and value types).

### `static` Members

Belong to the class, not any instance. Can be accessed without an object.

```cpp
class Box {
public:
    static int count;
    static void reset_count() { count = 0; }
};

int Box::count = 0; // definition must appear outside the class
```

Static member functions can only access static data members and other static member functions.

### `this` Pointer

Pointer to the current object, implicitly passed to all non-static member functions.

```cpp
class Box {
    int length, width, height;
public:
    Box& set_length(int l) { length = l; return *this; }
    Box& set_width(int w)  { width  = w; return *this; }
    Box& set_height(int h) { height = h; return *this; }
};

int main() {
    Box box(3, 4, 5);
    box.set_length(6).set_width(7).set_height(8); // method chaining
    return 0;
}
```

### `explicit` Keyword

Prevents the compiler from using a constructor for implicit type conversions.
```cpp
class MyInt {
public:
    explicit MyInt(int n) {}
};

void foo(MyInt m) {}

int main() {
    foo(42);        // Error: implicit conversion not allowed
    foo(MyInt(42)); // OK: explicit construction
    return 0;
}
```

## Principles

### Abstraction

Hide complex implementation details; expose only the essential interface.

### Encapsulation

Bundle data and the functions that operate on it into a single unit; restrict direct access to internal state.

### Inheritance

A derived class acquires attributes and behaviors of a base class.

```cpp
class Derived : public Base {};
```

**Inheritance access rules:**

| Base member | `public` inheritance | `protected` inheritance | `private` inheritance |
| ----------- | -------------------- | ----------------------- | --------------------- |
| `public`    | `public`             | `protected`             | `private`             |
| `protected` | `protected`          | `protected`             | `private`             |
| `private`   | inaccessible         | inaccessible            | inaccessible          |

**Multiple inheritance:**
```cpp
class Derived : public BaseA, public BaseB {};
```

Challenges:
- **Ambiguity**: two base classes share a member name — resolve with scope resolution `BaseA::method()`
- **Diamond problem**: two base classes both inherit from a common base — solve with virtual inheritance

```cpp
class A {};
class B : virtual public A {};
class C : virtual public A {};
class D : public B, public C {}; // only one copy of A
```

**Multilevel inheritance:**
```cpp
class Animal {};
class Dog : public Animal {};
class GoldenRetriever : public Dog {};
```

### Polymorphism

Objects of different types can be used through a common interface; behavior adapts based on the actual type at runtime.

**Virtual functions and late binding:**

`virtual` enables **dynamic dispatch** — the call is resolved at runtime based on the actual object type, not the pointer/reference type. Without `virtual`, the base class version is always called (static dispatch).

`= 0` makes a virtual function **pure virtual**, meaning it has no implementation in the base class. Derived classes **must** override it. A class with at least one pure virtual function is **abstract** and cannot be instantiated.

```cpp
class Shape {
public:
    virtual double area() const = 0; // pure virtual: makes Shape abstract
    virtual ~Shape() {}              // virtual destructor is essential for correct cleanup
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    double area() const override { return 3.14159 * radius * radius; }
};

int main() {
    Shape* s = new Circle(5.0);
    s->area(); // calls Circle::area at runtime (late binding)
    delete s;
}
```

- `override`: compile error if no matching virtual function exists in the base class
- `final`: prevents a virtual function from being overridden further, or prevents a class from being inherited. Also a performance hint — the compiler can devirtualize the call since no further overrides exist.
```cpp
class Circle : public Shape {
    void draw() override final {} // no subclass can override draw
};
class Leaf final : public Shape { // no class can inherit from Leaf
    void draw() override {}
};
```

**Object slicing**: assigning a derived object to a base object **by value** copies only the base portion — derived members and overridden behavior are lost. Use pointers or references to preserve polymorphism.
```cpp
Dog d;
Base b = d;    // sliced: only Base part copied, Dog part gone
b.speak();     // calls Base::speak

Base& ref = d;
ref.speak();   // calls Dog::speak — no slicing
```

# Exception Handling

```cpp
#include <stdexcept>

try {
    if (error_condition)
        throw std::runtime_error("something went wrong");
} catch (const std::runtime_error& e) {
    std::cerr << e.what();
} catch (const std::exception& e) {
    // catches any std::exception
} catch (...) {
    // catches everything
}
```

Common standard exceptions: `std::runtime_error`, `std::logic_error`, `std::out_of_range`, `std::invalid_argument`, `std::bad_alloc`.

`noexcept`: promises a function will not throw, allowing compiler optimizations.
```cpp
void foo() noexcept {}
```

# Resource Acquisition Is Initialization (RAII)

An object acquires a resource in its constructor and releases it in its destructor. This guarantees cleanup even when exceptions occur.

```cpp
class FileHandle {
    FILE* file;
public:
    FileHandle(const char* name) {
        file = fopen(name, "r");
        if (!file) throw std::runtime_error("Cannot open file");
    }
    ~FileHandle() {
        if (file) fclose(file);
    }
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};

void read_file() {
    FileHandle fh("data.txt"); // resource acquired
    // ... use fh ...
} // fh destructor called here — file always closed, even if an exception occurs
```

Smart pointers (`unique_ptr`, `shared_ptr`) are the standard RAII wrappers for heap memory.

# Templates

Generic programming: write functions and classes that work with any type.

## Function Templates

```cpp
template<typename T>
T max_val(T a, T b) {
    return (a > b) ? a : b;
}

max_val(1, 2);     // T = int
max_val(1.0, 2.0); // T = double
```

## Class Templates

```cpp
template<typename T>
class Stack {
    std::vector<T> data;
public:
    void push(const T& val) { data.push_back(val); }
    void pop()              { data.pop_back(); }
    T&   top()              { return data.back(); }
    bool empty() const      { return data.empty(); }
};

Stack<int> s;
s.push(1);
```

## Template Specialization

Override the generic template with a custom implementation for a specific type — like function overloading but for templates. The compiler picks the specialization when the type matches, otherwise uses the generic version.
```cpp
template<typename T>
T zero() { return T(0); } // generic: works for int, double, etc.

template<>
std::string zero<std::string>() { return ""; } // std::string(0) doesn't make sense, so provide a special case

zero<int>();         // uses generic → 0
zero<std::string>(); // uses specialization → ""
```

## Non-type Template Parameters

```cpp
template<typename T, int N>
class Array {
    T data[N];
};

Array<int, 10> arr;
```

## Variadic Templates

```cpp
template<typename... Args>
void print(Args... args) {
    ((std::cout << args << ' '), ...); // fold expression (C++17)
}

print(1, 2.5, "hello");
```

## Concepts (C++20)

Constrain template parameters:
```cpp
#include <concepts>

template<typename T>
requires std::integral<T> // only accepts types that support integral operation
T square(T x) { return x * x; }

// Shorthand
template<std::integral T>
T cube(T x) { return x * x * x; }
```

# Multithreading (`<thread>`)

**Concurrency**: managing overlapping tasks (not necessarily simultaneously)\
**Parallelism**: executing tasks simultaneously on multiple cores

```cpp
#include <iostream>
#include <thread>

void print(int i) {
    std::cout << i << '\n';
}

int main() {
    std::thread t(print, 0);
    t.join(); // wait for t to finish; or t.detach() to run independently
    return 0;
}
```

## Thread Synchronization

### Mutex (`<mutex>`)

Allows only one thread to access a shared resource at a time.
```cpp
#include <mutex>

std::mutex mtx;

void safe_print(int i) {
    std::lock_guard<std::mutex> lock(mtx); // RAII lock
    std::cout << i << '\n';
} // lock released automatically
```

| Type           | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `mutex`        | basic mutual exclusion                                               |
| `shared_mutex` | reader/writer lock (multiple readers, one writer)                    |
| `lock_guard`   | RAII wrapper; locks on construction, unlocks on destruction          |
| `unique_lock`  | like `lock_guard` but supports deferred locking, timed locking, and ownership transfer |
| `scoped_lock`  | locks multiple mutexes at once, deadlock-free (C++17)                |

### Semaphore (`<semaphore>`, C++20)

Use semaphores for signaling between threads; use mutexes for protecting shared data.
```cpp
#include <semaphore>

std::counting_semaphore<5> sem(5); // up to 5 concurrent accesses
std::binary_semaphore signal(0);   // for signaling between threads

sem.acquire(); // P operation (decrement; blocks if 0)
sem.release(); // V operation (increment)
```

### Condition Variable (`<condition_variable>`)

Lets a thread wait until another thread notifies it that a condition is true.
```cpp
#include <condition_variable>
#include <mutex>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void worker() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return ready; }); // releases lock while waiting; reacquires on notify
    // ... do work ...
}

void producer() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one(); // or notify_all()
}
```

### Atomic (`<atomic>`)

Lock-free operations on simple types; avoids data races without a mutex.
```cpp
#include <atomic>

std::atomic<int> counter(0);

void increment() {
    ++counter; // atomic read-modify-write
}
```

### Spin Lock

Busy-waits rather than sleeping. Low latency for very short critical sections; wastes CPU if held long.
```cpp
#include <atomic>

class SpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock()   { while (flag.test_and_set(std::memory_order_acquire)) {} }
    void unlock() { flag.clear(std::memory_order_release); }
};
```

# STL

## Containers

| Category              | Containers                                                                 |
| --------------------- | -------------------------------------------------------------------------- |
| Simple                | `pair`, `tuple`                                                            |
| Sequence              | `vector`, `deque`, `list` (doubly linked), `forward_list` (singly linked), `array` |
| Adapters              | `stack`, `queue`, `priority_queue`                                         |
| Ordered associative   | `set`, `map`, `multiset`, `multimap`                                       |
| Unordered associative | `unordered_set`, `unordered_map`, `unordered_multiset`, `unordered_multimap` |

```cpp
// Convert map to vector of pairs
std::vector<std::pair<int, int>> vec(umap.begin(), umap.end());
```

## Algorithms (`<algorithm>`)

**Non-modifying**: `for_each`, `any_of`, `all_of`, `none_of`, `count`, `find`, `search`\
**Modifying**: `copy`, `transform`, `fill`, `remove`, `unique`, `reverse`, `shuffle`, `swap`

**Sorting**: `sort`, `stable_sort`, `partial_sort`, `nth_element`\
**Binary search** (on sorted ranges): `binary_search`, `lower_bound`, `upper_bound`, `equal_range`

**Heap**: `make_heap`, `push_heap`, `pop_heap`, `sort_heap`\
**Set operations** (on sorted ranges): `set_union`, `set_intersection`, `set_difference`

**Numeric** (`<numeric>`): `iota`, `accumulate`, `inner_product`, `adjacent_difference`, `partial_sum`

See [cppreference](https://en.cppreference.com/w/cpp/algorithm)

## Iterators

Iterators abstract pointer-like access to container elements.

| Category          | Supports                       | Example containers         |
| ----------------- | ------------------------------ | -------------------------- |
| Input             | single-pass read               | `istream_iterator`         |
| Output            | single-pass write              | `ostream_iterator`         |
| Forward           | multi-pass read/write, `++`    | `forward_list`             |
| Bidirectional     | forward + `--`                 | `list`, `set`, `map`       |
| Random access     | bidirectional + `+n`, `-n`, `[]` | `vector`, `deque`, `array` |
| Contiguous (C++20)| random access + contiguous memory | `vector`, `array`       |

```cpp
std::vector<int> v = {1, 2, 3};

auto it = v.begin();  // iterator to first element
auto end = v.end();   // past-the-end iterator

++it;        // advance
*it;         // dereference
v.rbegin();  // reverse iterator (points to last element)
v.cbegin();  // const iterator

for (auto it = v.begin(); it != v.end(); ++it) {
    std::cout << *it;
}
```

## Functors (Function Objects)

A class with `operator()` overloaded, making instances callable. Can be inlined by the compiler, unlike function pointers.

```cpp
struct Multiplier {
    int factor;
    Multiplier(int f) : factor(f) {}
    int operator()(int x) const { return x * factor; }
};

Multiplier triple(3);
triple(5); // 15

// Used with algorithms
std::vector<int> v = {1, 2, 3, 4, 5};
std::transform(v.begin(), v.end(), v.begin(), Multiplier(2));
// v = {2, 4, 6, 8, 10}
```

Standard library functors in `<functional>`: `std::plus`, `std::minus`, `std::multiplies`, `std::divides`, `std::less`, `std::greater`, `std::negate`, etc.

```cpp
std::sort(v.begin(), v.end(), std::greater<int>()); // sort descending
```
