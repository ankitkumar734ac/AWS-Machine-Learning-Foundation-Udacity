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
### Splitting your Dataset
The first step in model training is to randomly split the dataset. This allows you to keep some data hidden during training, so that data can be used to evaluate your model before you put it into production. Specifically, you do this to test against the bias-variance trade-off. If you're interested in learning more, see the Further learning and reading section.

Splitting your dataset gives you two sets of data:

+ Training dataset: The data on which the model will be trained. Most of your data will be here. Many developers estimate about 80%.
+ Test dataset: The data withheld from the model during training, which is used to test how well your model will generalize to new data.
# Model Training Terminology
    The model training algorithm iteratively updates a model's parameters to minimize some loss function.
+ <b>Model parameters:</b> Model parameters are settings or configurations the training algorithm can update to change how the model behaves. Depending on the context, you’ll also hear other more specific terms used to describe model parameters such as weights and biases. Weights, which are values that change as the model learns, are more specific to neural networks.
+ <b>Loss function:</b> A loss function is used to codify the model’s distance from this goal. For example, if you were trying to predict a number of snow cone sales based on the day’s weather, you would care about making predictions that are as accurate as possible. So you might define a loss function to be “the average distance between your model’s predicted number of snow cone sales and the correct number.” You can see in the snow cone example this is the difference between the two purple dots.
## Putting it All Together
The end-to-end training process is
+ Feed the training data into the model.
+ Compute the loss function on the results.
+ Update the model parameters in a direction that reduces loss.
You continue to cycle through these steps until you reach a predefined stop condition. This might be based on a training time, the number of training cycles, or an even more intelligent or application-aware mechanism.
<b>Hyperparameters <b/> are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.
+ A loss function is used to codify the model’s distance from this goal
+ Training dataset: The data on which the model will be trained. Most of your data will be here.
+ Test dataset: The data withheld from the model during training, which is used to test how well your model will generalize to new data.
+ Model parameters are settings or configurations the training algorithm can update to change how the model behaves.

 # Step Four: Model Evaluation
 After you have collected your data and trained a model, you can start to evaluate how well your model is performing. The metrics used for evaluation are likely to be very specific to the problem you have defined. As you grow in your understanding of machine learning, you will be able to explore a wide variety of metrics that can enable you to evaluate effectively.
 ## Using Model Accuracy
Model accuracy is a fairly common evaluation metric. Accuracy is the fraction of predictions a model gets right.
 ![flowers](https://user-images.githubusercontent.com/71343747/124062811-079e0480-da4f-11eb-9a2a-c98692e2fdc6.png)
Imagine that you built a model to identify a flower as one of two common species based on measurable details like petal length. You want to know how often your model predicts the correct species. This would require you to look at your model's accuracy.
## Extended Learning
This information hasn't been covered in the above video but is provided for the advanced reader.

## Using Log Loss
Log loss seeks to calculate how uncertain your model is about the predictions it is generating. In this context, uncertainty refers to how likely a model thinks the predictions being generated are to be correct.
![jackets](https://user-images.githubusercontent.com/71343747/124062846-21d7e280-da4f-11eb-9aa8-ef8207d36252.png)
For example, let's say you're trying to predict how likely a customer is to buy either a jacket or t-shirt.

Log loss could be used to understand your model's uncertainty about a given prediction. In a single instance, your model could predict with 5% certainty that a customer is going to buy a t-shirt. In another instance, your model could predict with 80% certainty that a customer is going to buy a t-shirt. Log loss enables you to measure how strongly the model believes that its prediction is accurate.

In both cases, the model predicts that a customer will buy a t-shirt, but the model's certainty about that prediction can change.
 ## Remember: This Process is Iterative
![stepsiter](https://user-images.githubusercontent.com/71343747/124062928-564b9e80-da4f-11eb-8ad0-18483b81e651.png)
Every step we have gone through is highly iterative and can be changed or re-scoped during the course of a project. At each step, you might find that you need to go back and reevaluate some assumptions you had in previous steps. Don't worry! This ambiguity is normal.
# Step Five: Model Inference
Once you have trained your model, have evaluated its effectiveness, and are satisfied with the results, you're ready to generate predictions on real-world problems using unseen data in the field. In machine learning, this process is often called inference.
# Note
 Through the remainder of the lesson, we will be walking through 3 different case study examples of machine learning tasks actually solving problems in the real world.

 ## Supervised learning
Using machine learning to predict housing prices in a neighborhood based on lot size and number of bedrooms
## Unsupervised learning
Using machine learning to isolate micro-genres of books by analyzing the wording on the back cover description.
## Deep neural network
While this type of task is beyond the scope of this lesson, we wanted to show you the power and versatility of modern machine learning. You will see how it can be used to analyze raw images from lab video footage from security cameras, trying to detect chemical spills.
# Glossary


Bag of words: A technique used to extract features from the text. It counts how many times a word appears in a document (corpus), and then transforms that information into a dataset.

A categorical label has a discrete set of possible values, such as "is a cat" and "is not a cat."

Clustering. Unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.

CNN: Convolutional Neural Networks (CNN) represent nested filters over grid-organized data. They are by far the most commonly used type of model when processing images.

A continuous (regression) label does not have a discrete set of possible values, which means possibly an unlimited number of possibilities.

Data vectorization: A process that converts non-numeric data into a numerical format so that it can be used by a machine learning model.

Discrete: A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week).

FFNN: The most straightforward way of structuring a neural network, the Feed Forward Neural Network (FFNN) structures neurons in a series of layers, with each neuron in a layer containing weights to all neurons in the previous layer.

Hyperparameters are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.

Log loss is used to calculate how uncertain your model is about the predictions it is generating.

Hyperplane: A mathematical term for a surface that contains more than two planes.

Impute is a common term referring to different statistical tools which can be used to calculate missing values from your dataset.

label refers to data that already contains the solution.

loss function is used to codify the model’s distance from this goal

Machine learning, or ML, is a modern software development technique that enables computers to solve problems by using examples of real-world data.

Model accuracy is the fraction of predictions a model gets right. Discrete: A term taken from statistics referring to an outcome taking on only a finite number of values (such as days of the week). Continuous: Floating-point values with an infinite range of possible values. The opposite of categorical or discrete values, which take on a limited number of possible values.

Model inference is when the trained model is used to generate predictions.

model is an extremely generic program, made specific by the data used to train it.

Model parameters are settings or configurations the training algorithm can update to change how the model behaves.

Model training algorithms work through an interactive process where the current model iteration is analyzed to determine what changes can be made to get closer to the goal. Those changes are made and the iteration continues until the model is evaluated to meet the goals.

Neural networks: a collection of very simple models connected together. These simple models are called neurons. The connections between these models are trainable model parameters called weights.

Outliers are data points that are significantly different from others in the same sample.

Plane: A mathematical term for a flat surface (like a piece of paper) on which two points can be joined by a straight line.

Regression: A common task in supervised machine learning.

In reinforcement learning, the algorithm figures out which actions to take in a situation to maximize a reward (in the form of a number) on the way to reaching a specific goal.

RNN/LSTM: Recurrent Neural Networks (RNN) and the related Long Short-Term Memory (LSTM) model types are structured to effectively represent for loops in traditional computing, collecting state while iterating over some object. They can be used for processing sequences of data.

Silhouette coefficient: A score from -1 to 1 describing the clusters found during modeling. A score near zero indicates overlapping clusters, and scores less than zero indicate data points assigned to incorrect clusters. A

Stop words: A list of words removed by natural language processing tools when building your dataset. There is no single universal list of stop words used by all-natural language processing tools.

In supervised learning, every training sample from the dataset has a corresponding label or output value associated with it. As a result, the algorithm learns to predict labels or output values.

Test dataset: The data withheld from the model during training, which is used to test how well your model will generalize to new data.

Training dataset: The data on which the model will be trained. Most of your data will be here.

Transformer: A more modern replacement for RNN/LSTMs, the transformer architecture enables training over larger datasets involving sequences of data.

In unlabeled data, you don't need to provide the model with any kind of label or solution while the model is being trained.

In unsupervised learning, there are no labels for the training data. A machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.




































