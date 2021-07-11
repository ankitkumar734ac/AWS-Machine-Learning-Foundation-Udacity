# Software Engineering Practices
## Clean and Modular Code
+ <b>Production code:</b> Software running on production servers to handle live users and data of the intended audience. Note that this is different from production-quality code, which describes code that meets expectations for production in reliability, efficiency, and other aspects. Ideally, all code in production meets these expectations, but this is not always the case.
+ <b>Clean code:</b> Code that is readable, simple, and concise. Clean production-quality code is crucial for collaboration and maintainability in software development.
+ <b>Modular code:</b> Code that is logically broken up into functions and modules. Modular production-quality code that makes your code more organized, efficient, and reusable.
+ <b>Module:</b> A file. Modules allow code to be reused by encapsulating them into files that can be imported into other files.
## Refactoring Code
+ Refactoring: Restructuring your code to improve its internal structure without changing its external functionality. This gives you a chance to clean and modularize your program after you've got it working.
+ Since it isn't easy to write your best code while you're still trying to just get it working, allocating time to do this is essential to producing high-quality code. Despite the initial time and effort required, this really pays off by speeding up your development time in the long run.
+ You become a much stronger programmer when you're constantly looking to improve your code. The more you refactor, the easier it will be to structure and write good code the first time.
## Writing clean code: Meaningful names
Use meaningful names.

+ Be descriptive and imply type: For booleans, you can prefix with is_ or has_ to make it clear it is a condition. You can also use parts of speech to imply types, like using verbs for functions and nouns for variables.
+ Be consistent but clearly differentiate: age_list and age is easier to differentiate than ages and age.
+ Avoid abbreviations and single letters: You can determine when to make these exceptions based on the audience for your code. If you work with other data scientists, certain variables may be common knowledge. While if you work with full stack engineers, it might be necessary to provide more descriptive names in these cases as well. (Exceptions include counters and common math variables.)
+ Long names aren't the same as descriptive names: You should be descriptive, but only with relevant information. For example, good function names describe what they do well without including details about implementation or highly specific uses.
## Writing clean code: Nice whitespace
Use whitespace properly.

+ Organize your code with consistent indentation: the standard is to use four spaces for each indent. You can make this a default in your text editor.
+ Separate sections with blank lines to keep your code well organized and readable.
+ Try to limit your lines to around 79 characters, which is the guideline given in the PEP 8 style guide. In many good text editors, there is a setting to display a subtle line that indicates where the 79 character limit is.

#### check out the code layout section of PEP 8 in the following notes.
```sh
# Correct:

# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)
```
```sh
# Wrong:

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```
```sh
# Optional:
# Hanging indents *may* be indented to other than 4 spaces.
foo = long_function_name(
  var_one, var_two,
  var_three, var_four)
```
# Writing Modular Code
Follow the tips below to write modular code.

+ Tip: DRY (Don't Repeat Yourself)
Don't repeat yourself! Modularization allows you to reuse parts of your code. Generalize and consolidate repeated code in functions or loops.

+ Tip: Abstract out logic to improve readability
Abstracting out code into a function not only makes it less repetitive, but also improves readability with descriptive function names. Although your code can become more readable when you abstract out logic into functions, it is possible to over-engineer this and have way too many modules, so use your judgement.

+ Tip: Minimize the number of entities (functions, classes, modules, etc.)
There are trade-offs to having function calls instead of inline logic. If you have broken up your code into an unnecessary amount of functions and modules, you'll have to jump around everywhere if you want to view the implementation details for something that may be too small to be worth it. Creating more modules doesn't necessarily result in effective modularization.

+ Tip: Functions should do one thing
Each function you write should be focused on doing one thing. If a function is doing multiple things, it becomes more difficult to generalize and reuse. Generally, if there's an "and" in your function name, consider refactoring.

+ Tip: Arbitrary variable names can be more effective in certain functions
Arbitrary variable names in general functions can actually make the code more readable.

+ Tip: Try to use fewer than three arguments per function
Try to use no more than three arguments when possible. This is not a hard rule and there are times when it is more appropriate to use many parameters. But in many cases, it's more effective to use fewer arguments. Remember we are modularizing to simplify our code and make it more efficient. If your function has a lot of parameters, you may want to rethink how you are splitting this up.

# Efficient Code
Knowing how to write code that runs efficiently is another essential skill in software development. Optimizing code to be more efficient can mean making it:

Execute faster
Take up less space in memory/storage
The project on which you're working determines which of these is more important to optimize for your company or product. When you're performing lots of different transformations on large amounts of data, this can make orders of magnitudes of difference in performance.
# Documentation
 Documentation: Additional text or illustrated information that comes with or is embedded in the code of software.
+ Documentation is helpful for clarifying complex parts of code, making your code easier to navigate, and quickly conveying how and why different components of your program are used.
 Several types of documentation can be added at different levels of your program:
+ Inline comments - line level
+ Docstrings - module and function level
+ Project documentation - project level

# Inline Comments
+ Inline comments are text following hash symbols throughout your code. They are used to explain parts of your code, and really help future contributors understand your work.
+ Comments often document the major steps of complex code. Readers may not have to understand the code to follow what it does if the comments explain it. However, others would argue that this is using comments to justify bad code, and that if code requires comments to follow, it is a sign refactoring is needed.
+ Comments are valuable for explaining where code cannot. For example, the history behind why a certain method was implemented a specific way. Sometimes an unconventional or seemingly arbitrary approach may be applied because of some obscure external variable causing side effects. These things are difficult to explain with code.
# Docstrings
Docstring, or documentation strings, are valuable pieces of documentation that explain the functionality of any function or module in your code. Ideally, each of your functions should always have a docstring.

# Project Documentation
Project documentation is essential for getting others to understand why and how your code is relevant to them, whether they are potentials users of your project or developers who may contribute to your code. A great first step in project documentation is your README file. It will often be the first interaction most users will have with your project.

# Version Control In Data Science
If you need a refresher on using Git for version control, check out the course linked in the extracurriculars. If you're ready, let's see how Git is used in real data science scenarios!
## Scenario #1
Let's walk through the Git commands that go along with each step in the scenario you just observed in the video.

Step 1: You have a local version of this repository on your laptop, and to get the latest stable version, you pull from the develop branch.
Switch to the develop branch
git checkout develop

Pull the latest changes in the develop branch
git pull

Step 2: When you start working on this demographic feature, you create a new branch called demographic, and start working on your code in this branch.
Create and switch to a new branch called demographic from the develop branch
git checkout -b demographic

Work on this new feature and commit as you go
git commit -m 'added gender recommendations'
git commit -m 'added location specific recommendations'
...

Step 3: However, in the middle of your work, you need to work on another feature. So you commit your changes on this demographic branch, and switch back to the develop branch.
Commit your changes before switching
git commit -m 'refactored demographic gender and location recommendations '

Switch to the develop branch
git checkout develop

Step 4: From this stable develop branch, you create another branch for a new feature called friend_groups.
Create and switch to a new branch called friend_groups from the develop branch
git checkout -b friend_groups

Step 5: After you finish your work on the friend_groups branch, you commit your changes, switch back to the development branch, merge it back to the develop branch, and push this to the remote repository’s develop branch.
Commit your changes before switching
git commit -m 'finalized friend_groups recommendations '

Switch to the develop branch
git checkout develop

Merge the friend_groups branch into the develop branch
git merge --no-ff friends_groups

Push to the remote repository
git push origin develop

Step 6: Now, you can switch back to the demographic branch to continue your progress on that feature.
Switch to the demographic branch
git checkout demographic
## Scenario #2
Let's walk through the Git commands that go along with each step in the scenario you just observed in the video.

Step 1: You check your commit history, seeing messages about the changes you made and how well the code performed.
View the log history
git log

Step 2: The model at this commit seemed to score the highest, so you decide to take a look.
Check out a commit
git checkout bc90f2cbc9dc4e802b46e7a153aa106dc9a88560

After inspecting your code, you realize what modifications made it perform well, and use those for your model.

Step 3: Now, you're confident merging your changes back into the development branch and pushing the updated recommendation engine.
Switch to the develop branch
git checkout develop

Merge the friend_groups branch into the develop branch
git merge --no-ff friend_groups

Push your changes to the remote repository
git push origin develop

## Scenario #3
Let's walk through the Git commands that go along with each step in the scenario you just observed in the video.

Step 1: Andrew commits his changes to the documentation branch, switches to the development branch, and pulls down the latest changes from the cloud on this development branch, including the change I merged previously for the friends group feature.
Commit the changes on the documentation branch
git commit -m "standardized all docstrings in process.py"

Switch to the develop branch
git checkout develop

Pull the latest changes on the develop branch down
git pull

Step 2: Andrew merges his documentation branch into the develop branch on his local repository, and then pushes his changes up to update the develop branch on the remote repository.
Merge the documentation branch into the develop branch
git merge --no-ff documentation

Push the changes up to the remote repository
git push origin develop

Step 3: After the team reviews your work and Andrew's work, they merge the updates from the development branch into the master branch. Then, they push the changes to the master branch on the remote repository. These changes are now in production.
Merge the develop branch into the master branch
git merge --no-ff develop

Push the changes up to the remote repository
git push origin master
<hr>
<br>
# Welcome To Software Engineering Practices, Part 2
In part 2 of software engineering practices, you'll learn about the following practices of software engineering and how they apply in data science.

Testing
Logging
Code reviews

# Testing
Testing your code is essential before deployment. It helps you catch errors and faulty conclusions before they make any major impact. Today, employers are looking for data scientists with the skills to properly prepare their code for an industry setting, which includes testing their code.
# Testing And Data Science
+ Problems that could occur in data science aren’t always easily detectable; you might have values being encoded incorrectly, features being used inappropriately, or unexpected data breaking assumptions.
+ To catch these errors, you have to check for the quality and accuracy of your analysis in addition to the quality of your code. Proper testing is necessary to avoid unexpected surprises and have confidence in your results.
+ Test-driven development (TDD): A development process in which you write tests for tasks before you even write the code to implement those tasks.
+ Unit test: A type of test that covers a “unit” of code—usually a single function—independently from the rest of the program.
# Unit tests
We want to test our functions in a way that is repeatable and automated. Ideally, we'd run a test program that runs all our unit tests and cleanly lets us know which ones failed and which ones succeeded. Fortunately, there are great tools available in Python that we can use to create effective unit tests!

Unit test advantages and disadvantages
The advantage of unit tests is that they are isolated from the rest of your program, and thus, no dependencies are involved. They don't require access to databases, APIs, or other external sources of information. However, passing unit tests isn’t always enough to prove that our program is working successfully. To show that all the parts of our program work with each other properly, communicating and transferring data between them correctly, we use integration tests. In this lesson, we'll focus on unit tests; however, when you start building larger programs, you will want to use integration tests as well.
# Unit Testing Tools
To install pytest, run pip install -U pytest in your terminal. You can see more information on getting started here.

Create a test file starting with test_.
Define unit test functions that start with test_ inside the test file.
Enter pytest into your terminal in the directory of your test file and it detects these tests for you.
test_ is the default; if you wish to change this, you can learn how in this pytest configuration.

In the test output, periods represent successful unit tests and Fs represent failed unit tests. Since all you see is which test functions failed, it's wise to have only one assert statement per test. Otherwise, you won't know exactly how many tests failed or which tests failed.

Your test won't be stopped by failed assert statements, but it will stop if you have syntax errors.

# Test-driven development and data science
Test-driven development: Writing tests before you write the code that’s being tested. Your test fails at first, and you know you’ve finished implementing a task when the test passes.
Tests can check for different scenarios and edge cases before you even start to write your function. When start implementing your function, you can run the test to get immediate feedback on whether it works or not as you tweak your function.
When refactoring or adding to your code, tests help you rest assured that the rest of your code didn't break while you were making those changes. Tests also helps ensure that your function behavior is repeatable, regardless of external parameters such as hardware and time.
# Logging
Logging is valuable for understanding the events that occur while running your program. For example, if you run your model overnight and the results the following morning are not what you expect, log messages can help you understand more about the context in those results occurred. Let's learn about the qualities that make a log message effective.

# Log messages
Logging is the process of recording messages to describe events that have occurred while running your software. Let's take a look at a few examples, and learn tips for writing good log messages.

Tip: Be professional and clear
```sh
Bad: Hmmm... this isn't working???
Bad: idk.... :(
Good: Couldn't parse file.
```
Tip: Be concise and use normal capitalization
```sh
Bad: Start Product Recommendation Process
Bad: We have completed the steps necessary and will now proceed with the recommendation process for the records in our product database.
Good: Generating product recommendations.
```
Tip: Choose the appropriate level for logging
```sh
Debug: Use this level for anything that happens in the program. Error: Use this level to record any error that occurs. Info: Use this level to record all actions that are user driven or system specific, such as regularly scheduled operations.
```
Tip: Provide any useful information
```sh
Bad: Failed to read location data
Good: Failed to read location data: store_id 8324971
```
# Code reviews
Code reviews benefit everyone in a team to promote best programming practices and prepare code for production. Let's go over what to look for in a code review and some tips on how to conduct one.
# Questions to ask yourself when conducting a code review
First, let's look over some of the questions we might ask ourselves while reviewing code. These are drawn from the concepts we've covered in these last two lessons.

+ Is the code clean and modular?
Can I understand the code easily?
Does it use meaningful names and whitespace?
+ Is there duplicated code?
Can I provide another layer of abstraction?
Is each function and module necessary?
Is each function or module too long?
+ Is the code efficient?
Are there loops or other steps I can vectorize?
Can I use better data structures to optimize any steps?
Can I shorten the number of calculations needed for any steps?
Can I use generators or multiprocessing to optimize any steps?
+ Is the documentation effective?
Are inline comments concise and meaningful?
Is there complex code that's missing documentation?
Do functions use effective docstrings?
Is the necessary project documentation provided?
+ Is the code well tested?
Does the code high test coverage?
Do tests check for interesting cases?
Are the tests readable?
Can the tests be made more efficient?
+ Is the logging effective?
Are log messages clear, concise, and professional?
Do they include all relevant and useful information?
Do they use the appropriate logging level?


































