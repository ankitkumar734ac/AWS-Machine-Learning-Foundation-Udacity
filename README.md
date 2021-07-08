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
<b>Hyperparameters </b> are settings on the model which are not changed during training but can affect how quickly or how reliably the model trains, such as the number of clusters the model should identify.
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

# Machine Learning with AWS
### Why AWS?
The AWS achine learning mission is to put machine learning in the hands of every developer.
+ AWS offers the broadest and deepest set of artificial intelligence (AI) and machine learning (ML) services with unmatched flexibility.
+ You can accelerate your adoption of machine learning with AWS SageMaker. Models that previously took months to build and required specialized expertise can now be built in weeks or even days.
+ AWS offers the most comprehensive cloud offering optimized for machine learning.
+ More machine learning happens at AWS than anywhere else.
# AWS Machine Learning offerings
## AWS AI services
By using AWS pre-trained AI services, you can apply ready-made intelligence to a wide range of applications such as personalized recommendations, modernizing your contact center, improving safety and security, and increasing customer engagement.

## Industry-specific solutions
With no knowledge in machine learning needed, add intelligence to a wide range of applications in different industries including healthcare and manufacturing.
## AWS Machine Learning services
With AWS, you can build, train, and deploy your models fast. Amazon SageMaker is a fully managed service that removes complexity from ML workflows so every developer and data scientist can deploy machine learning for a wide range of use cases.
## ML infrastructure and frameworks
AWS Workflow services make it easier for you to manage and scale your underlying ML infrastructure.
![screen-shot-2021-04-19-at-10 54 32-am](https://user-images.githubusercontent.com/71343747/124373967-bdb65800-dcb4-11eb-9991-8e626a2afd4e.png)
## AWS DeepLens:
A deep learning–enabled video camera
## AWS DeepRacer:
An autonomous race car designed to test reinforcement learning models by racing on a physical track
## AWS DeepComposer: 
A composing device powered by generative AI that creates a melody that transforms into a completely original song
## AWS ML Training and Certification: 
Curriculum used to train Amazon developers
# AWS Account Requirements
# Train your computer vision model with AWS DeepLens (optional)
To train and deploy custom models to AWS DeepLens, you use Amazon SageMaker. Amazon SageMaker is a separate service and has its own service pricing and billing tier. It's not required to train a model for this course. If you're interested in training a custom model, please note that it incurs a cost. To learn more about SageMaker costs, see the Amazon SageMaker Pricing.
# Train your reinforcement learning model with AWS DeepRacer
To get started with AWS DeepRacer, you receive 10 free hours to train or evaluate models and 5GB of free storage during your first month. This is enough to train your first time-trial model, evaluate it, tune it, and then enter it into the AWS DeepRacer League. This offer is valid for 30 days after you have used the service for the first time.
Beyond 10 hours of training and evaluation, you pay for training, evaluating, and storing your machine learning models. Charges are based on the amount of time you train and evaluate a new model and the size of the model stored. To learn more about AWS DeepRacer pricing, see the AWS DeepRacer Pricing
# Generate music using AWS DeepComposer
To get started, AWS DeepComposer provides a 12-month Free Tier for first-time users. With the Free Tier, you can perform up to 500 inference jobs translating to 500 pieces of music using the AWS DeepComposer Music studio. You can use one of these instances to complete the exercise at no cost. To learn more about AWS DeepComposer costs, see the AWS DeepComposer Pricing.
# Build a custom generative AI model (GAN) using Amazon SageMaker (optional)
Amazon SageMaker is a separate service and has its own service pricing and billing tier. To train the custom generative AI model, the instructor uses an instance type that is not covered in the Amazon SageMaker free tier. If you want to code along with the instructor and train your own custom model, you may incur a cost. Please note, that creating your own custom model is completely optional. You are not required to do this exercise to complete the course. To learn more about SageMaker costs, see the Amazon SageMaker Pricing.
## Computer Vision and Its Applications
This section introduces you to common concepts in computer vision (CV), and explains how you can use AWS DeepLens to start learning with computer vision projects. By the end of this section, you will be able to explain how to create, train, deploy, and evaluate a trash-sorting project that uses AWS DeepLens.
Modern-day applications of computer vision use neural networks. These networks can quickly be trained on millions of images and produce highly accurate predictions.
#How computer vision got started
+ Early applications of computer vision needed hand-annotated images to successfully train a model.
+ These early applications had limited applications because of the human labor required to annotate images.
# Three main components of neural networks
+ Input Layer: This layer receives data during training and when inference is performed after the model has been trained.
+ Hidden Layer: This layer finds important features in the input data that have predictive power based on the labels provided during training.
+ Output Layer: This layer generates the output or prediction of your model.
#Modern computer vision
+ Modern-day applications of computer vision use neural networks call convolutional neural networks or CNNs.
+ In these neural networks, the hidden layers are used to extract different information about images. We call this process feature extraction.
+ These models can be trained much faster on millions of images and generate a better prediction than earlier models.
#How this growth occured
Since 2010, we have seen a rapid decrease in the computational costs required to train the complex neural networks used in computer vision.
Larger and larger pre-labeled datasets have become generally available. This has decreased the time required to collect the data needed to train many models.
# Computer Vision Applications
Computer vision (CV) has many real-world applications. In this video, we cover examples of image classification, object detection, semantic segmentation, and activity recognition. Here's a brief summary of what you learn about each topic in the video:

+ Image classification is the most common application of computer vision in use today. Image classification can be used to answer questions like What's in this image? This type of task has applications in text detection or optical character recognition (OCR) and content moderation.
+ Object detection is closely related to image classification, but it allows users to gather more granular detail about an image. For example, rather than just knowing whether an object is present in an image, a user might want to know if there are multiple instances of the same object present in an image, or if objects from different classes appear in the same image.
+ Semantic segmentation is another common application of computer vision that takes a pixel-by-pixel approach. Instead of just identifying whether an object is present or not, it tries to identify down the pixel level which part of the image is part of the object.
+ Activity recognition is an application of computer vision that is based around videos rather than just images. Video has the added dimension of time and, therefore, models are able to detect changes that occur over time.
<br> <hr>
# Computer Vision with AWS DeepLens
## AWS DeepLens
AWS DeepLens allows you to create and deploy end-to-end computer vision–based applications. The following video provides a brief introduction to how AWS DeepLens works and how it uses other AWS services.
AWS DeepLens is a deep learning–enabled camera that allows you to deploy trained models directly to the device. You can either use sample templates and recipes or train your own model.
AWS DeepLens is integrated with several AWS machine learning services and can perform local inference against deployed models provisioned from the AWS Cloud. It enables you to learn and explore the latest artificial intelligence (AI) tools and techniques for developing computer vision applications based on a deep learning model.
# How AWS DeepLens works
AWS DeepLens is integrated with multiple AWS services. You use these services to create, train, and launch your AWS DeepLens project. You can think of an AWS DeepLens project as being divided into two different streams as the image shown above.
First, you use the AWS console to create your project, store your data, and train your model.
Then, you use your trained model on the AWS DeepLens device. On the device, the video stream from the camera is processed, inference is performed, and the output from inference is passed into two output streams:
+ Device stream – The video stream passed through without processing.
+ Project stream – The results of the model's processing of the video frames.
<img width="495" alt="screen-shot-2021-04-16-at-2 36 08-pm" src="https://user-images.githubusercontent.com/71343747/124374225-30283780-dcb7-11eb-942b-9fd3d9cebf97.png">

# A Sample Project with AWS DeepLens
This section provides a hands-on demonstration of a project created as part of an AWS DeepLens sponsored hack-a-thon. In this project, we use an AWS DeepLens device to do an image classification–based task. We train a model to detect if a piece of trash is from three potential classes: landfill, compost, or recycling.
AWS DeepLens is integrated with multiple AWS services. You use these services to create, train, and launch your AWS DeepLens project. To create any AWS DeepLens–based project you will need an AWS account.
Four key components are required for an AWS DeepLens–based project.
+ Collect your data: Collect data and store it in an Amazon S3 bucket.
+ Train your model: Use a Jupyter Notebook in Amazon SageMaker to train your model.
+ Deploy your model: Use AWS Lambda to deploy the trained model to your AWS DeepLens device.
+ View model output: Use Amazon IoT Greenrass to view your model's output after the model is deployed.
# Demo: Using the AWS console to set up and deploy an AWS DeepLens project
[aws-deeplens-custom-trash-detector.ipynb.txt](https://github.com/ankitkumar734ac/AWS-Machine-Learning-Foundation-Udacity/files/6759430/aws-deeplens-custom-trash-detector.ipynb.txt)

# Reinforcement Learning and Its Applications
This section introduces you to a type of machine learning (ML) called reinforcement learning (RL). You'll hear about its real-world applications and learn basic concepts using AWS DeepRacer as an example. By the end of the section, you will be able to create, train, and evaluate a reinforcement learning model in the AWS DeepRacer console.

In reinforcement learning (RL), an agent is trained to achieve a goal based on the feedback it receives as it interacts with an environment. It collects a number as a reward for each action it takes. Actions that help the agent achieve its goal are incentivized with higher numbers. Unhelpful actions result in a low reward or no reward.

With a learning objective of maximizing total cumulative reward, over time, the agent learns, through trial and error, to map gainful actions to situations. The better trained the agent, the more efficiently it chooses actions that accomplish its goal.
# Reinforcement Learning Applications
Reinforcement learning is used in a variety of fields to solve real-world problems. It’s particularly useful for addressing sequential problems with long-term goals. Let’s take a look at some examples.

+ RL is great at playing games:
++ Go (board game) was mastered by the AlphaGo Zero software.
++ Atari classic video games are commonly used as a learning tool for creating and testing RL software.
++ StarCraft II, the real-time strategy video game, was mastered by the AlphaStar software.
+ RL is used in video game level design:
++ Video game level design determines how complex each stage of a game is and directly affects how boring, frustrating, or fun it is to play that game.
++ Video game companies create an agent that plays the game over and over again to collect data that can be visualized on graphs.
++ This visual data gives designers a quick way to assess how easy or difficult it is for a player to make progress, which enables them to find that “just right” balance between boredom and frustration faster.
+ RL is used in wind energy optimization:
++ RL models can also be used to power robotics in physical devices.
++ When multiple turbines work together in a wind farm, the turbines in the front, which receive the wind first, can cause poor wind conditions for the turbines behind them. ++ This is called wake turbulence and it reduces the amount of energy that is captured and converted into electrical power.
++ Wind energy organizations around the world use reinforcement learning to test solutions. Their models respond to changing wind conditions by changing the angle of the turbine blades. When the upstream turbines slow down it helps the downstream turbines capture more energy.
+ Other examples of real-world RL include:
++ Industrial robotics
++ Fraud detection
++ Stock trading
++ Autonomous driving

# Reinforcement Learning Concepts
This section introduces six basic reinforcement learning terms and provides an example for each in the context of AWS DeepRacer.
![l3-ml-with-aws-rl-terms](https://user-images.githubusercontent.com/71343747/124374558-e260fe80-dcb9-11eb-8693-1d330ed36c58.png)
## Agent
+ The piece of software you are training is called an agent.
+ It makes decisions in an environment to reach a goal.
+ In AWS DeepRacer, the agent is the AWS DeepRacer car and its goal is to finish * laps around the track as fast as it can while, in some cases, avoiding obstacles.
## Environment
+ The environment is the surrounding area within which our agent interacts.
+ For AWS DeepRacer, this is a track in our simulator or in real life.
## State
+ The state is defined by the current position within the environment that is visible, or known, to an agent.
+ In AWS DeepRacer’s case, each state is an image captured by its camera.
+ The car’s initial state is the starting line of the track and its terminal state is when the car finishes a lap, bumps into an obstacle, or drives off the track.
## Action
+ For every state, an agent needs to take an action toward achieving its goal.
+ An AWS DeepRacer car approaching a turn can choose to accelerate or brake and turn left, right, or go straight.
## Reward
+ Feedback is given to an agent for each action it takes in a given state.
+ This feedback is a numerical reward.
+ A reward function is an incentive plan that assigns scores as rewards to different zones on the track.
##Episode
+ An episode represents a period of trial and error when an agent makes decisions and gets feedback from its environment.
+ For AWS DeepRacer, an episode begins at the initial state, when the car leaves the starting position, and ends at the terminal state, when it finishes a lap, bumps into an obstacle, or drives off the track.
In a reinforcement learning model, an agent learns in an interactive real-time environment by trial and error using feedback from its own actions. Feedback is given in the form of rewards.
# Putting Your Spin on AWS DeepRacer: The Practitioner's Role in RL
AWS DeepRacer may be autonomous, but you still have an important role to play in the success of your model. In this section, we introduce the training algorithm, action space, hyperparameters, and reward function and discuss how your ideas make a difference.

An algorithm is a set of instructions that tells a computer what to do. ML is special because it enables computers to learn without being explicitly programmed to do so.
The training algorithm defines your model’s learning objective, which is to maximize total cumulative reward. Different algorithms have different strategies for going about this.
A soft actor critic (SAC) embraces exploration and is data-efficient, but can lack stability.
A proximal policy optimization (PPO) is stable but data-hungry.
An action space is the set of all valid actions, or choices, available to an agent as it interacts with an environment.
Discrete action space represents all of an agent's possible actions for each state in a finite set of steering angle and throttle value combinations.
Continuous action space allows the agent to select an action from a range of values that you define for each sta te.
Hyperparameters are variables that control the performance of your agent during training. There is a variety of different categories with which to experiment. Change the values to increase or decrease the influence of different parts of your model.
For example, the learning rate is a hyperparameter that controls how many new experiences are counted in learning at each step. A higher learning rate results in faster training but may reduce the model’s quality.
The reward function's purpose is to encourage the agent to reach its goal. Figuring out how to reward which actions is one of your most important jobs.
# Putting Reinforcement Learning into Action with AWS DeepRacer
This video put the concepts we've learned into action by imagining the reward function as a grid mapped over the race track in AWS DeepRacer’s training environment, and visualizing it as metrics plotted on a graph. It also introduced the trade-off between exploration and exploitation, an important challenge unique to this type of machine learning.
<img width="689" alt="screen-shot-2021-04-27-at-3 44 12-pm" src="https://user-images.githubusercontent.com/71343747/124374604-471c5900-dcba-11eb-944a-62d0f0a12237.png">
Each square is a state. The green square is the starting position, or initial state, and the finish line is the goal, or terminal state.

Key points to remember about reward functions:

Each state on the grid is assigned a score by your reward function. You incentivize behavior that supports your car’s goal of completing fast laps by giving the highest numbers to the parts of the track on which you want it to drive.
The reward function is the actual code you'll write to help your agent determine if the action it just took was good or bad, and how good or bad it was.
<img width="696" alt="screen-shot-2021-04-27-at-3 46 11-pm" src="https://user-images.githubusercontent.com/71343747/124374615-5e5b4680-dcba-11eb-9ec2-8da3af1a7ec3.png">
Key points to remember about exploration versus exploitation:

When a car first starts out, it explores by wandering in random directions. However, the more training an agent gets, the more it learns about an environment. This experience helps it become more confident about the actions it chooses.
Exploitation means the car begins to exploit or use information from previous experiences to help it reach its goal. Different training algorithms utilize exploration and exploitation differently.
Key points to remember about the reward graph:

While training your car in the AWS DeepRacer console, your training metrics are displayed on a reward graph.
Plotting the total reward from each episode allows you to see how the model performs over time. The more reward your car gets, the better your model performs.
Key points to remember about AWS DeepRacer:

AWS DeepRacer is a combination of a physical car and a virtual simulator in the AWS Console, the AWS DeepRacer League, and community races.
An AWS DeepRacer device is not required to start learning: you can start now in the AWS console. The 3D simulator in the AWS console is where training and evaluation take place.
# Exercise: Interpret the reward graph of your first AWS DeepRacer model
Instructions
Train a model in the AWS DeepRacer console and interpret its reward graph.

Part 1: Train a reinforcement learning model using the AWS DeepRacer console
Practice the knowledge you've learned by training your first reinforcement learning model using the AWS DeepRacer console.

If this is your first time using AWS DeepRacer, choose Get started from the service landing page, or choose Get started with reinforcement learning from the main navigation pane.
On the Get started with reinforcement learning page, under Step 2: Create a model and race, choose Create model. Alternatively, on the AWS DeepRacer home page, choose Your models from the main navigation pane to open the Your models page. On the Your models page, choose Create model.
On the Create model page, under Environment simulation, choose a track as a virtual environment to train your AWS DeepRacer agent. Then, choose Next. For your first run, choose a track with a simple shape and smooth turns. In later iterations, you can choose more complex tracks to progressively improve your models. To train a model for a particular racing event, choose the track most similar to the event track.
On the Create model page, choose Next.
On the Create Model page, under Race type, choose a training type. For your first run, choose Time trial. The agent with the default sensor configuration with a single-lens camera is suitable for this type of racing without modifications.
On the Create model page, under Training algorithm and hyperparameters, choose the Soft Actor Critic (SAC) or Proximal Policy Optimization (PPO) algorithm. In the AWS DeepRacer console, SAC models must be trained in continuous action spaces. PPO models can be trained in either continuous or discrete action spaces.
On the Create model page, under Training algorithm and hyperparameters, use the default hyperparameter values as is. Later on, to improve training performance, expand the hyperparameters and experiment with modifying the default hyperparameter values.
On the Create model page, under Agent, choose The Original DeepRacer or The Original DeepRacer (continuous action space) for your first model. If you use Soft Actor Critic (SAC) as your training algorithm, we filter your cars so that you can conveniently choose from a selection of compatible continuous action space agents.
On the Create model page, choose Next.
On the Create model page, under Reward function, use the default reward function example as is for your first model. Later on, you can choose Reward function examples to select another example function and then choose Use code to accept the selected reward function.
On the Create model page, under Stop conditions, leave the default Maximum time value as is or set a new value to terminate the training job to help prevent long-running (and possible run-away) training jobs. When experimenting in the early phase of training, you should start with a small value for this parameter and then progressively train for longer amounts of time.
On the Create model page, choose Create model to start creating the model and provisioning the training job instance.
After the submission, watch your training job being initialized and then run. The initialization process takes about 6 minutes to change status from Initializing to In progress.
Watch the Reward graph and Simulation video stream to observe the progress of your training job. You can choose the refresh button next to Reward graph periodically to refresh the Reward graph until the training job is complete.
Note: The training job is running on the AWS Cloud, so you don't need to keep the AWS DeepRacer console open during training. However, you can come back to the console to check on your model at any point while the job is in progress.

Part 2: Inspect your reward graph to assess your training progress
As you train and evaluate your first model, you'll want to get a sense of its quality. To do this, inspect your reward graph.

Find the following on your reward graph:

Average reward
Average percentage completion (training)
Average percentage completion (evaluation)
Best model line
Reward primary y-axis
Percentage track completion secondary y-axis
Iteration x-axis
Review the solution to this exercise for ideas on how to interpret it.

<img width="404" alt="best-model-bar-reward-graph2" src="https://user-images.githubusercontent.com/71343747/124374764-a9298e00-dcbb-11eb-9939-1d44d3b1bb9d.png">

Exercise Solution
To get a sense of how well your training is going, watch the reward graph. Here is a list of its parts and what they do:

Average reward
This graph represents the average reward the agent earns during a training iteration. The average is calculated by averaging the reward earned across all episodes in the training iteration. An episode begins at the starting line and ends when the agent completes one loop around the track or at the place the vehicle left the track or collided with an object. Toggle the switch to hide this data.
Average percentage completion (training)
The training graph represents the average percentage of the track completed by the agent in all training episodes in the current training. It shows the performance of the vehicle while experience is being gathered.
Average percentage completion (evaluation)
While the model is being updated, the performance of the existing model is evaluated. The evaluation graph line is the average percentage of the track completed by the agent in all episodes run during the evaluation period.
Best model line
This line allows you to see which of your model iterations had the highest average progress during the evaluation. The checkpoint for this iteration will be stored. A checkpoint is a snapshot of a model that is captured after each training (policy-updating) iteration.
Reward primary y-axis
This shows the reward earned during a training iteration. To read the exact value of a reward, hover your mouse over the data point on the graph.
Percentage track completion secondary y-axis
This shows you the percentage of the track the agent completed during a training iteration.
Iteration x-axis
This shows the number of iterations completed during your training job.

<img width="404" alt="best-model-bar-reward-graph2 (1)" src="https://user-images.githubusercontent.com/71343747/124374784-c0687b80-dcbb-11eb-8e7d-74f435d388ed.png">

Reward Graph Interpretation
The following four examples give you a sense of how to interpret the success of your model based on the reward graph. Learning to read these graphs is as much of an art as it is a science and takes time, but reviewing the following four examples will give you a start.

Needs more training
In the following example, we see there have only been 600 iterations, and the graphs are still going up. We see the evaluation completion percentage has just reached 100%, which is a good sign but isn’t fully consistent yet, and the training completion graph still has a ways to go. This reward function and model are showing promise, but need more training time.

<img width="457" alt="udacity-reward-graph-needs-more-training" src="https://user-images.githubusercontent.com/71343747/124374792-cb231080-dcbb-11eb-8b86-e330aed8dec2.png">

No improvement
In the next example, we can see that the percentage of track completions haven’t gone above around 15 percent and it's been training for quite some time—probably around 6000 iterations or so. This is not a good sign! Consider throwing this model and reward function away and trying a different strategy.

<img width="456" alt="udacity-reward-graph-bad-graph" src="https://user-images.githubusercontent.com/71343747/124374799-d83fff80-dcbb-11eb-86c7-c9fa4f5f7f5a.png">

A well-trained model
In the following example graph, we see the evaluation percentage completion reached 100% a while ago, and the training percentage reached 100% roughly 100 or so iterations ago. At this point, the model is well trained. Training it further might lead to the model becoming overfit to this track.



Avoid overfitting
Overfitting or overtraining is a really important concept in machine learning. With AWS DeepRacer, this can become an issue when a model is trained on a specific track for too long. A good model should be able to make decisions based on the features of the road, such as the sidelines and centerlines, and be able to drive on just about any track.

An overtrained model, on the other hand, learns to navigate using landmarks specific to an individual track. For example, the agent turns a certain direction when it sees uniquely shaped grass in the background or a specific angle the corner of the wall makes. The resulting model will run beautifully on that specific track, but perform badly on a different virtual track, or even on the same track in a physical environment due to slight variations in angles, textures, and lighting.

<img width="424" alt="udacity-reward-graph-overfitting" src="https://user-images.githubusercontent.com/71343747/124374814-e68e1b80-dcbb-11eb-80d4-0575aa1f6f23.png">

Adjust hyperparameters
The AWS DeepRacer console's default hyperparameters are quite effective, but occasionally you may consider adjusting the training hyperparameters. The hyperparameters are variables that essentially act as settings for the training algorithm that control the performance of your agent during training. We learned, for example, that the learning rate controls how many new experiences are counted in learning at each step.
In this reward graph example, the training completion graph and the reward graph are swinging high and low. This might suggest an inability to converge, which may be helped by adjusting the learning rate. Imagine if the current weight for a given node is .03, and the optimal weight should be .035, but your learning rate was set to .01. The next training iteration would then swing past optimal to .04, and the following iteration would swing under it to .03 again. If you suspect this, you can reduce the learning rate to .001. A lower learning rate makes learning take longer but can help increase the quality of your model.

<img width="415" alt="udacity-reward-graph-adjust-hyperparameters" src="https://user-images.githubusercontent.com/71343747/124374826-f574ce00-dcbb-11eb-9a65-e6739f91e85d.png">

Good Job and Good Luck!
Remember: training experience helps both model and reinforcement learning practitioners become a better team. Enter your model in the monthly AWS DeepRacer League races for chances to win prizes and glory while improving your machine learning development skills!


#  Introduction to Generative AI
Generative AI is one of the biggest recent advancements in artificial intelligence because of its ability to create new things.

Until recently, the majority of machine learning applications were powered by discriminative models. A discriminative model aims to answer the question, "If I'm looking at some data, how can I best classify this data or predict a value?" For example, we could use discriminative models to detect if a camera was pointed at a cat.

As we train this model over a collection of images (some of which contain cats and others which do not), we expect the model to find patterns in images which help make this prediction.

A generative model aims to answer the question,"Have I seen data like this before?" In our image classification example, we might still use a generative model by framing the problem in terms of whether an image with the label "cat" is more similar to data you’ve seen before than an image with the label "no cat."

However, generative models can be used to support a second use case. The patterns learned in generative models can be used to create brand new examples of data which look similar to the data it seen before.

<img width="778" alt="screen-shot-2021-05-04-at-12 55 15-pm" src="https://user-images.githubusercontent.com/71343747/124845005-61ed1700-dfb3-11eb-9dc0-a077e759ef44.png">

# Generative AI Models
## Autoregressive models
Autoregressive convolutional neural networks (AR-CNNs) are used to study systems that evolve over time and assume that the likelihood of some data depends only on what has happened in the past. It’s a useful way of looking at many systems, from weather prediction to stock prediction.

##vGenerative adversarial networks (GANs)
Generative adversarial networks (GANs), are a machine learning model format that involves pitting two networks against each other to generate new content. The training algorithm swaps back and forth between training a generator network (responsible for producing new data) and a discriminator network (responsible for measuring how closely the generator network’s data represents the training dataset).

## Transformer-based models
Transformer-based models are most often used to study data with some sequential structure (such as the sequence of words in a sentence). Transformer-based methods are now a common modern tool for modeling natural language.

# What is AWS DeepComposer?
AWS DeepComposer gives you a creative and easy way to get started with machine learning (ML), specifically generative AI. It consists of a USB keyboard that connects to your computer to input melody and the AWS DeepComposer console, which includes AWS DeepComposer Music studio to generate music, learning capsules to dive deep into generative AI models, and AWS DeepComposer Chartbusters challenges to showcase your ML skills.
# What are GANs?
A GAN is a type of generative machine learning model which pits two neural networks against each other to generate new content: a generator and a discriminator.

A generator is a neural network that learns to create new data resembling the source data on which it was trained.
A discriminator is another neural network trained to differentiate between real and synthetic data.
The generator and the discriminator are trained in alternating cycles. The generator learns to produce more and more realistic data while the discriminator iteratively gets better at learning to differentiate real data from the newly created data.
# Generator: 
A neural network that learns to create new data resembling the source data on which it was trained.
# Discriminator: 
A neural network trained to differentiate between real and synthetic data.
# Generator loss: 
Measures how far the output data deviates from the real data present in the training dataset.
# Discriminator loss:
Evaluates how well the discriminator differentiates between real and fake data.
# AR-CNN with AWS DeepComposer
When a note is either added or removed from your input track during inference, we call it an edit event. To train the AR-CNN model to predict when notes need to be added or removed from your input track (edit event), the model iteratively updates the input track to sounds more like the training dataset. During training, the model is also challenged to detect differences between an original piano roll and a newly modified piano roll.

New Terms
Piano roll: A two-dimensional piano roll matrix that represents input tracks. Time is on the horizontal axis and pitch is on the vertical axis.
Edit event: When a note is either added or removed from your input track during inference.
<img width="1291" alt="aws-mle-demo-gan-image" src="https://user-images.githubusercontent.com/71343747/124845850-366b2c00-dfb5-11eb-8298-68789767ab4a.png">










