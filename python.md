Python
======

# Terminology

`Compiled`: source code is translated to machine code or bytecode before execution, resulting in an executable\
`Interpreted`: source code is translated line by line at runtime
- Property of the implementation, not the language itself (eg. Python is compiled to bytecode -> PVM can execute bytecode with interpreter or using Just-In-Time compilation)

`Strongly` vs `weakly`: how strict types are enforced (eg. int + str allowed?)

`Static` vs `dynamic`: when types are checked (compile time or runtime)

`Pass by value`: function receives a copy of the variable value\
`Pass by reference`: function receives a reference to the variable

# Built-in Types

## Truth Value Testing
An object is evaluated to `True` unless the class has a `__bool__()` method that returns `False` or `__len__()` that returns 0. Objects considered `False`:
- Constants with value `None` or `False`
- `0`, `0.0`, `0j`
- `''`, `()`, `[]`, `{}`, `set()`, `range(0)`

## Bit Manipulation
| Operation | Symbol |
| - | - |
| Bitwise AND | a & b |
| Bitwise OR | a \| b |
| Bitwise XOR | a ^ b |
| Bitwise NOT | ~a |
| Bitwise left shift | a << b |
| Bitwise right shift | a >> b |

## Float, Complex
Constructor
- `float()` accepts `'inf'` and `'nan'` (optional prefix: +/-)
- `complex(real, imag)`

Scientific notation: `3e4`, `1.4e-2`

Complex: `1 + 2j`
- `z.real` = 1.0, `z.imag` = 2.0

## Binary
Conversion
- `bin(x)`: convert int to binary string prefixed with '0b'
- `int(x, 2)`: convert binary string to int

## Strings
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

## F-string
`f'{a = }, {b = }, {c = }'`\
`f'{a} + {b} = {c}'` equivalent to `f'{a + b = }'`

| Format | Syntax | Result (`num = 10`) |
| - | - | - |
| 2 decimal places | `{num:.2f}` | 10.00
| Hex | `{num:#x}` | 0xa |
| Binary | `{num:b}` | 1010 |
| Scientic notation | `{num:e}` | 1.0e+1 |
| 5 digits | `{num:05}` | 00010 |

## Print & I/O
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

# Iterator
Iterator vs Iterable: Objects like strings and lists are iterable, but not iterators

## Turn iterable into iterator
`__iter__(obj)` returns iterator object\
`__iter__(obj, sentinel)`
- `obj` must be callable (check out [Objects](#objects))
- `obj` will be called until `sentinel` value is returned
```python
s = 'hello'
s = iter(s)
print(next(s))
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
    num = yield # get input from outside
    while True:
        num = yield num + 1
sg = stateful_generator()
next(sg) # start generator
sg.send(10)
sg.send(12)
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

## Lambda functions
- Anonymous: don't require name
- Short-term use
```python
x = lambda a: a + 1
print(x(5))

y = lambda a, b: a + b
print(y(3, 4))
```

## Conditional statements
`map`

`filter`

`reduce`

## Variable Scope
A function can read a global variable but can't modify it
```python
x = 1

def foo():
    print(x)    # reads global x
    x = 2       # new local x, doesn't modify global x

def bar():
    x = x + 1   # error: local x referenced before assignment
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

## Main function
```python
def main():
    ...

if __name__ == "__main__":
    main()
```

## Functions are Objects
Everything in Python is an object, including functions\
A callable is an object that can be called using `()`, ie. functions, objects, classes. `()` invokes `__call__()`

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

# Class
Class variable: shared across all instances\
Instance variable: unique to each instance

Protected member: denoted by single underscore, should be accessed by class and subclass\
Private member: denoted by double underscore, should be accessed by class

`self`: Python converts obj.method(args) to ClassName.method(obj, args), so self is required to specify which instance to call on

`@staticmethod`: doesn't depend on object, callable without instantiating the class\
`@classmethod`: also callable without instantiating, but follows subclass via inheritence, not super class
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

## Inheritance
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

## Aditional Class Methods
`id(obj)`: unique id of object

`getattr(obj, attr: str)`: returns obj.attr, error if attr does not exist\
`getattr(obj, attr: str, default)`: returns default if attr does not exist

`setattr(obj, attr: str, value)`

## Type vs Instance
`isinstance()` supports inheritance, `type()` does not

```python
type(student) == Person     # False
isinstance(student, Person) # True
```

# Multithreading
Global interpreter lock (GIL)
