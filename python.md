# Python

# Terminology

`Compiled`: source code is translated to machine code or bytecode before execution, resulting in an executable\
`Interpreted`: source code is translated line by line at runtime
- Property of the implementation, not the language itself (e.g. Python is compiled to bytecode → the PVM executes bytecode with an interpreter or using Just-In-Time compilation)

`Strongly` vs `weakly typed`: how strictly types are enforced (e.g. is `int + str` allowed?)

`Static` vs `dynamic typed`: when types are checked (compile time vs runtime)

`Pass by value`: function receives a copy of the argument\
`Pass by reference`: function receives a reference to the original variable

# Built-in Types

## Truth Value Testing

An object is evaluated to `True` unless its class defines `__bool__()` that returns `False`, or `__len__()` that returns `0`. Objects considered `False`:
- Constants with value `None` or `False`
- `0`, `0.0`, `0j`
- `''`, `()`, `[]`, `{}`, `set()`, `range(0)`

## Bit Manipulation

| Operation           | Symbol   |
| ------------------- | -------- |
| Bitwise AND         | `a & b`  |
| Bitwise OR          | `a \| b` |
| Bitwise XOR         | `a ^ b`  |
| Bitwise NOT         | `~a`     |
| Bitwise left shift  | `a << b` |
| Bitwise right shift | `a >> b` |

## Float, Complex

Constructors:
- `float()` accepts `'inf'` and `'nan'` (optional prefix: `+`/`-`)
- `complex(real, imag)`

Scientific notation: `3e4`, `1.4e-2`

Complex: `z = 1 + 2j`
- `z.real` = `1.0`, `z.imag` = `2.0`

## Binary

Conversion:
- `bin(x)`: convert int to binary string prefixed with `'0b'`
- `int(x, 2)`: convert binary string to int

## Strings

`chr(97)` = `'a'`\
`ord('a')` = `97`

`s.find(value, start, stop)`: first occurrence of `value`, returns `-1` if not found. `rfind` for last occurrence.

Check if string is a number:
- `s.isdecimal()`
- `s.isdigit()`
- `s.isnumeric()`

Check if string is alphanumeric:
- `s.isalnum()`

`s.strip(characters)` (default characters = whitespace)
- e.g. `s.strip(',.')` to remove any leading/trailing commas and periods
- `lstrip` for leading and `rstrip` for trailing

`s.lower()`, `s.upper()`

`s.split()`, `' '.join(list)`

## F-string

`f'{a = }, {b = }, {c = }'` prints each name and its value (e.g. `a = 1, b = 2, c = 3`).\
`f'{a + b = }'` evaluates the expression and prints `a + b = 3`.

| Format              | Syntax       | Result (`num = 10`) |
| ------------------- | ------------ | ------------------- |
| 2 decimal places    | `{num:.2f}`  | `10.00`             |
| Hex                 | `{num:#x}`   | `0xa`               |
| Binary              | `{num:b}`    | `1010`              |
| Scientific notation | `{num:e}`    | `1.0e+1`            |
| 5 digits            | `{num:05}`   | `00010`             |

## Print & I/O

`print(*obj, sep=' ', end='\n')`

| Mode                | Symbol | If file doesn't exist | If file exists  |
| ------------------- | ------ | --------------------- | --------------- |
| Read only (default) | `'r'`  | Error                 |                 |
| Write only          | `'w'`  | Create                | Delete + create |
| Append              | `'a'`  | Create                |                 |
| Create              | `'x'`  | Create                | Error           |

| Mode                 | Symbol |
| -------------------- | ------ |
| Text (default)       | `'t'`  |
| Binary (e.g. images) | `'b'`  |

```python
with open('file.ext', 'r') as f:
    data = f.read()
```

# Iterator

Iterator vs Iterable: objects like strings and lists are iterable, but not iterators themselves.

## Turn iterable into iterator

`iter(obj)` returns an iterator object.\
`iter(callable, sentinel)`:
- `callable` must be a zero-argument callable (see [Functions are Objects](#functions-are-objects))
- `callable` will be called until it returns the `sentinel` value

```python
s = 'hello'
it = iter(s)
print(next(it))
```

## Generator

```python
def generator():
    yield 1
    yield 2
    yield 3

g = generator()
print(next(g))
```

```python
def stateful_generator():
    num = yield          # get input from outside
    while True:
        num = yield num + 1

sg = stateful_generator()
next(sg)        # start generator
sg.send(10)     # yields 11
sg.send(12)     # yields 13
```

# Functions

## Type Hinting

```python
num: int = 0

def type_hinting(x: int, y: int | None = None) -> list[int]:
    ...
```

## Packing

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

## Lambda Functions

- Anonymous: don't require a name
- Short-term use

```python
x = lambda a: a + 1
print(x(5))

y = lambda a, b: a + b
print(y(3, 4))
```

## Functional Built-ins

`map(func, iterable)`: apply `func` to each element\
`filter(func, iterable)`: keep elements where `func` returns truthy\
`functools.reduce(func, iterable, initial)`: cumulatively apply `func` to reduce to a single value

```python
from functools import reduce

list(map(lambda x: x * 2, [1, 2, 3]))       # [2, 4, 6]
list(filter(lambda x: x > 1, [1, 2, 3]))    # [2, 3]
reduce(lambda a, b: a + b, [1, 2, 3], 0)    # 6
```

## Variable Scope

A function can read a global variable but cannot rebind it without `global`. Any assignment to a name inside a function makes that name local for the entire function body — referencing it before assignment raises `UnboundLocalError`.

```python
x = 1

def foo():
    print(x)    # UnboundLocalError: x is local because of the assignment below
    x = 2

def bar():
    x = x + 1   # UnboundLocalError: local x referenced before assignment
```

Use `global` so a local name refers to a module-level variable:

```python
x = 1

def foo():
    global x
    x = x + 1   # global x is modified

def bar():
    global y    # binds module-level y
    y = 2
```

Use `nonlocal` so an inner function refers to an enclosing function's scope:

```python
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 9
    inner()
    print(x)    # 9
```

## Main Function

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

## Functions are Objects

Everything in Python is an object, including functions.\
A callable is an object that can be invoked using `()` — e.g. functions, classes, and instances of classes that define `__call__`. `obj()` invokes `obj.__call__()`.

**Examples**
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

## Decorator

A function that takes another function as an argument and extends/modifies its behavior.

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

# Class

Class variable: shared across all instances\
Instance variable: unique to each instance

Protected member: denoted by a single underscore; convention says it should only be accessed by the class and its subclasses\
Private member: denoted by a double underscore; name-mangled to `_ClassName__name`

`self`: Python converts `obj.method(args)` to `ClassName.method(obj, args)`, so `self` is required to identify the instance the method is called on.

`@staticmethod`: doesn't depend on the instance or class; callable without instantiating\
`@classmethod`: also callable without instantiating; receives the class as the first argument (`cls`), which respects subclassing

```python
class Person:
    species = 'homo sapien'             # class variable

    def __init__(self, name='', age=1):
        self.name = name                # instance variable
        self.age = age

        self._ssn = 0                   # protected member

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

## Inheritance

```python
class Student(Person):
    def __init__(self, name='', age=1, student_id=None):
        super().__init__(name, age)     # same as Person.__init__(self, name, age)
        self.student_id = student_id

        self.__school = 'uw'            # private member (name-mangled to _Student__school)

person = Person('Optimus', 500)
student = Student('Prime', 10, 1)
baby = person + student
```

## Additional Class Methods

`id(obj)`: unique id of an object

`getattr(obj, attr: str)`: returns `obj.attr`, raises `AttributeError` if `attr` does not exist\
`getattr(obj, attr: str, default)`: returns `default` if `attr` does not exist

`setattr(obj, attr: str, value)`: equivalent to `obj.attr = value`

## Type vs Instance

`isinstance()` supports inheritance, `type()` does not.

```python
type(student) == Person         # False
isinstance(student, Person)     # True
```

# Multithreading

Global Interpreter Lock (GIL): in CPython, only one thread executes Python bytecode at a time. Threads are still useful for I/O-bound work; use `multiprocessing` for CPU-bound parallelism.
