Machine learning, or ML, is a modern software development technique that enables computers to solve problems by using examples of real-world data.

In supervised learning, every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values.

In reinforcement learning, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal.

In unsupervised learning, there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.

# Components of Machine Learning
![clay99](https://user-images.githubusercontent.com/71343747/123567249-53e30d80-d7df-11eb-9545-ab74a285313b.png)
Nearly all tasks solved with machine learning involve three primary components:
 + A machine learning model
 + A model training algorithm
 + A model inference algorithm
# What are machine learning models?
A machine learning model, like a piece of clay, can be molded into many different forms and serve many different purposes. A more technical definition would be that a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided.

### Important
     A model is an extremely generic program(or block of code), 
     made specific by the data used to train it. 
     It is used to solve different problems.
# Model
A model is an extremely generic program, made specific by the data used to train it.

# Model training algorithms
Model training algorithms work through an interactive process where the current model iteration is analyzed to determine what changes can be made to get closer to the goal. Those changes are made and the iteration continues until the model is evaluated to meet the goals.

# Model inference
Model inference is when the trained model is used to generate predictions.

![Screenshot (603)](https://user-images.githubusercontent.com/71343747/123568329-b3421d00-d7e1-11eb-9a0a-7ab6c72dc22e.png)

# Major Steps in the Machine Learning Process
![steps](https://user-images.githubusercontent.com/71343747/123568532-2a77b100-d7e2-11eb-8a32-9fed9a0e252e.png)

# What is a Machine Learning Task?
All model training algorithms, and the models themselves, take data as their input. Their outputs can be very different and are classified into a few different groups based on the task they are designed to solve. Often, we use the kind of data required to train a model as part of defining a machine learning task.

In this lesson, we will focus on two common machine learning tasks:
+ Supervised learning
+ Unsupervised learning
# Supervised and Unsupervised Learning
The presence or absence of labeling in your data is often used to identify a machine learning task.
![mltask](https://user-images.githubusercontent.com/71343747/123729641-f3250500-d8b2-11eb-925a-e6b608d0b6df.png)
# Supervised tasks
A task is supervised if you are using labeled data. We use the term labeled to refer to data that already contains the solutions, called labels.
    'For example: Predicting the number of snow cones sold based on the temperatures is an example of supervised learning.'

![snowcones2](https://user-images.githubusercontent.com/71343747/123729837-37b0a080-d8b3-11eb-9d50-0a6dcef17185.png)
In the preceding graph, the data contains both a temperature and the number of snow cones sold. Both components are used to generate the linear regression shown on the graph. Our goal was to predict the number of snow cones sold, and we feed that value into the model. We are providing the model with labeled data and therefore, we are performing a supervised machine learning task.

# Unsupervised tasks
A task is considered to be unsupervised if you are using unlabeled data. This means you don't need to provide the model with any kind of label or solution while the model is being trained.

Let's take a look at unlabeled data.
![tree2](https://user-images.githubusercontent.com/71343747/123730164-ab52ad80-d8b3-11eb-9964-d6a94fb7f69a.png)![tree](https://user-images.githubusercontent.com/71343747/123729919-56af3280-d8b3-11eb-8f7f-ea7d1fa4609f.png)
+ Take a look at the preceding picture. Did you notice the tree in the picture? What you just did, when you noticed the object in the picture and identified it as a tree, is called labeling the picture. Unlike you, a computer just sees that image as a matrix of pixels of varying intensity.
+ Since this image does not have the labeling in its original data, it is considered unlabeled.
# How do we classify tasks when we don't have a label?
Unsupervised learning involves using data that doesn't have a label. One common task is called clustering. Clustering helps to determine if there are any naturally occurring groupings in the data.
# Further Classifying by using Label Types
![snsupersuper](https://user-images.githubusercontent.com/71343747/123730376-dfc66980-d8b3-11eb-9461-a5d69ca789b8.png)
Initially, we divided tasks based on the presence or absence of labeled data while training our model. Often, tasks are further defined by the type of label which is present.
In <b>supervised</b> learning, there are two main identifiers you will see in machine learning:
+ A categorical label has a discrete set of possible values. In a machine learning problem in which you want to identify the type of flower based on a picture, you would train your model using images that have been labeled with the categories of flower you would want to identify. Furthermore, when you work with categorical labels, you often carry out classification tasks*, which are part of the supervised learning family.
+ A continuous (regression) label does not have a discrete set of possible values, which often means you are working with numerical data. In the snow cone sales example, we are trying to predict the number* of snow cones sold. Here, our label is a number that could, in theory, be any value.
In unsupervised learning, <b>clustering</b> is just one example. There are many other options, such as deep learning.
+ <b>Clustering.</b> Unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
+ A <b>categorical label</b> has a discrete set of possible values, such as "is a cat" and "is not a cat."
+ A <b>continuous (regression)</b> label does not have a discrete set of possible values, which means possibly an unlimited number of possibilities.
+ <b>Discrete: </b>A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).
+ A <b>label </b>refers to data that already contains the solution.
Using unlabeled data means you don't need to provide the model with any kind of label or solution while the model is being trained.


# Step Two: Build a Dataset
The Four Aspects of Working with Data
![datasteps](https://user-images.githubusercontent.com/71343747/123731003-f3be9b00-d8b4-11eb-99b4-40c9f30b5352.png)
# Data collection
Data collection can be as straightforward as running the appropriate SQL queries or as complicated as building custom web scraper applications to collect data for your project. You might even have to run a model over your data to generate needed labels. Here is the fundamental question:
Does the data you've collected match the machine learning task and problem you have define
# Data inspection
The quality of your data will ultimately be the largest factor that affects how well you can expect your model to perform. As you inspect your data, look for:OutliersMissing or incomplete values
Data that needs to be transformed or preprocessed so it's in the correct format to be used by your model
# Data visualization
You can use data visualization to see outliers and trends in your data and to help stakeholders understand your data.
Look at the following two graphs. In the first graph, some data seems to have clustered into different groups. In the second graph, some data points might be outliers.
![plot](https://user-images.githubusercontent.com/71343747/123731164-3f714480-d8b5-11eb-98e5-a10e213d00f3.png)
![plot2](https://user-images.githubusercontent.com/71343747/123731166-413b0800-d8b5-11eb-9e58-0cc801fb5802.png)
In machine learning, you use several statistical-based tools to better understand your data. The sklearn library has many examples and tutorials, such as this example demonstrating < a herf="https://sklearn.org/auto_examples/applications/plot_outlier_detection_housing.html#sphx-glr-auto-examples-applications-plot-outlier-detection-housing-py" >outlier detection on a real dataset.</a>
# Step Three: Model Training





