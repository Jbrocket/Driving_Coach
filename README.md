### Driving Coach

## Part 1
Description: I want to create a project that aims to develop a system
for real-time object detection and tracking for drivers while driving
down an ordinary road, bustling street, or jammed highway with the primary
goal of enhancing road safety and potentially traffic management/precaution
by identifying and monitoring multiple objects of interest simultaneously.
These objects include road signs, open road/your current lane, pedestrians,
lights, etc. 

To accomplish these goals, here are some high level considerations that
I am currently in the process of identifying and figuring out:

# Object Detection and Localization
We've done some of this in class, but being able to identify figures
of interest (as mentioned above) and putting boundary boxes around them
to demostrate that the system can keep track of objects and can discern them
from other objects of similar type. In addition to this, localization
entails distances which could be used to determine different things such as
braking time for the car that's needed in different terrains, conditions, and 
speeds.

# Object Tracking
This ties into the last one with locality, but being able to track objects and how
they are moving from the systems view so that objects of similar type
don't become entangled, making a more cohesive system

# Pixel Identification
Being able to use semantic segmentation by identifying pixels available on the
screen and confidently saying what that pixel belongs to in relation
to the objects that appear on the road such as street, signs, pedestrians,
etc. We should be able to know what a pixel belongs to.

# Object Recognition and Classification
Being able to tell what objects are and potentially using techniques that
are able to identify specific types of objects such as types of cars, types
of signs, etc. These are all important factors when driving on the road,
the system being able to discern what an object is as if we were identifying
it could play key roles in how it evaluates situations.

# Driver Assistance
These ties in everything previously into this, but using the collected data
and the models evaluation, it's able to make decisions (a.k.a. suggestions)
like braking when there's a stopped car or red light, speeding up when the flow
of traffic is picking up, swerving when there's not enough stopping distance to brake
fast enough for an object in front of you, and more.

# Datasets
There is data posted on the canvas for this project, and I think it should be sufficient 
enough to choose from. In addition, there should be a validation set to see how the model
reacts to new data and determine whether there should be adjustments in order to avoid
overfitting and underfitting for specific data.I just need a certain amount of 
data to train on, then I need data to demonstrate the model for testing that's separate 
from its training data. The initial dataset should include a wide variety of roads 
and conditions for the driver so that there is more to look at for the model. 

# Techniques for object detection
We've already gone through some basics of object detection such as tracking by
hue/color. In addition to this, we can detect objects through shape based detection
which could be really useful in tandem with color based to detect road signs and
road lanes. We can also use the key points of features such as corners, edges, unique
patterns, textures, etc that are specific to an object that will allow us to have a
better object tracking mechanism


## Part 2