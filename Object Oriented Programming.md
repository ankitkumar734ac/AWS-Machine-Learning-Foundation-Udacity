# Why object-oriented programming?
# Why object-oriented programming?
Object-oriented programming has a few benefits over procedural programming, which is the programming style you most likely first learned. As you'll see in this lesson:

+ Object-oriented programming allows you to create large, modular programs that can easily expand over time.
+ Object-oriented programs hide the implementation from the end user.
# Procedural versus object-oriented programming
Objects are defined by characteristics and actions
Here is a reminder of what is a characteristic and what is an action.
Characteristics and actions in English grammar
You can also think about characteristics and actions is in terms of English grammar. A characteristic corresponds to a noun and an action corresponds to a verb.

Let's pick something from the real world: a dog. Some characteristics of the dog include the dog's weight, color, breed, and height. These are all nouns. Some actions a dog can take include to bark, to run, to bite, and to eat. These are all verbs.
# Object-oriented programming (OOP) vocabulary
Class: A blueprint consisting of methods and attributes.
Object: An instance of a class. It can help to think of objects as something in the real world like a yellow pencil, a small dog, or a blue shirt. However, as you'll see later in the lesson, objects can be more abstract.
Attribute: A descriptor or characteristic. Examples would be color, length, size, etc. These attributes can take on specific values like blue, 3 inches, large, etc.
Method: An action that a class or object could take.
OOP: A commonly used abbreviation for object-oriented programming.
Encapsulation: One of the fundamental ideas behind object-oriented programming is called encapsulation: you can combine functions and data all into a single entity. In object-oriented programming, this single entity is called a class. Encapsulation allows you to hide implementation details, much like how the scikit-learn package hides the implementation of machine learning algorithms.
![screen-shot-2018-07-19-at-4 06 55-pm](https://user-images.githubusercontent.com/71343747/125601494-0b907ccd-e625-4327-9060-315e4cb212e9.png)
# Object-oriented programming syntax
A function and a method look very similar. They both use the def keyword. They also have inputs and return outputs. The difference is that a method is inside of a class whereas a function is outside of a class.

### What is self?
If you instantiate two objects, how does Python differentiate between these two objects?
```sh
shirt_one = Shirt('red', 'S', 'short-sleeve', 15)
shirt_two = Shirt('yellow', 'M', 'long-sleeve', 20)
```
That's where self comes into play. If you call the change_price method on shirt_one, how does Python know to change the price of shirt_one and not of shirt_two?
```sh
shirt_one.change_price(12)
```
Behind the scenes, Python is calling the change_price method:
```sh
    def change_price(self, new_price):

        self.price = new_price
 ```
Self tells Python where to look in the computer's memory for the shirt_one object. Then, Python changes the price of the shirt_one object. When you call the change_price method, shirt_one.change_price(12), self is implicitly passed in.

The word self is just a convention. You could actually use any other name as long as you are consisten, but you should use self to avoid confusing people.
# 





































