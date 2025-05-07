# SOLID Principles

## Single Responsibility
**"A class should have only one reason to change"**\
That is, each module should do one thing

## Open-Closed
**"Modules should be open for extension, but closed for modification"**

## Liskov Substitution
**"Derived or child classes must be substitutable for their base or parent classes"**

## Interface Segregation
**"Clients should not be forced to depend on interfaces it doesn't use"**\
That is, instead of one large interface, have multiple smaller interfaces

## Dependency Inversion
**"High-level modules should not depend on low-level modules. Both should depend on abstractions"**\
Additionally, abstractions should not depend on details. Details should depend on abstractions

# Design Patterns

## Creational Design Patterns
Abstract the instantiation process

### Factory Method
A method creates objects without specifying the exact class. This allows subclasses to override the factory method and change the return product class

Implementation
1. Create interface or abstract class
2. Implement the interface or extend the abstract class
3. Define creator class and declare the factory method
4. Define concrete creator classes and override factory method

<details>
<summary>Example</summary>

```
interface Button
    render()

class WindowsButton implements Button
    ...
class LinuxButton implements Button
    ...

class Dialog
    abstract Button createButton()

    render():
        Button b = createButton()
        b.render()

class WindowsDialog extends Dialog
    Button createButton(): return new WindowsButton()
class LinuxDialog extends Dialog
    Button createButton(): return new LinuxButton()

class Application
    Dialog dialog

    init()
        config = ...
        if config == Windows
            dialog = new WindowsDialog
        else if config == Linux
            dialog = new LinuxDialog
    main()
        init()
        dialog.render()
```
</details>

### Abstract Factory
An interface with multiple factory methods for creating related objects (common base class). Considered as another layer of abstraction over factory pattern

<details>
<summary>Example</summary>

```
interface GUIFactory
    createButton()
    createCheckbox()

class WindowsFactory implements GUIFactory
    ...
class LinuxFactory implements GUIFactory
    ...

interface Button
    render()

class WindowsButton implements Button
    ...
class LinuxButton implements Button
    ...

interface Checkbox
    render()

class WindowsCheckbox implements Checkbox
    ...
class LinuxCheckbox implements Checkbox
    ...

class Application
    GUIFactory factory
    Button button
    Checkbox checkbox
    Application(factory)
        this.factory = factory
    createUI()
        this.button = factory.createButton()
        this.checkbox = factory.createCheckbox()
    render()
        button.render()
        checkbox.render()

class ApplicationConfigurator
    main()
        config = ...
        if config == Windows
            factory = new WindowsFactory
        else if config == Linux
            factory = new LinuxFactory
        Application app = new Application(factory)
```
</details>

### Builder
Construct complex objects step by step.

If a class depends on many parameters, there will be too many subclasses to cover all combinations\
You could have a giant constructor in the base class with all possible parameters, but most parameters will go unused

Builder pattern extracts object construction code out of its own class and into separate objects called builders

### Prototype
Copy an existing object instead of creating a new instance from scratch

Implementation
1. Protoype interface declares clone()
2. Concrete protoype class implements clone()
3. Client can produce copies of any object that follows the protoype interface

### Singleton
A class only has one instance and provides global access point to it

Implementation
1. Make default constructor private to prevent it from being used
2. Create a static creation method that acts as a constructor. This method calls the private constructor, and all following method calls return the cached object

## Structural Design Patterns
Assemble objects and classes into larger structures

### Adapter
Allow objects with incompatible interfaces to interact

### Bridge

### Composite

### Decorator

### Facade

### Flyweight

### Proxy

## Behavorial Design Patterns
Algorithms and the assignment of responsibilities between object

### Chain of Responsibility

### Command

### Iterator

### Mediator

### Memento

### Observer

### State

### Strategy

### Template

### Visitor
