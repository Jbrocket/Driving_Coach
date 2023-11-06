# Driving Coach

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

### Object Detection and Localization
We've done some of this in class, but being able to identify figures
of interest (as mentioned above) and putting boundary boxes around them
to demostrate that the system can keep track of objects and can discern them
from other objects of similar type. In addition to this, localization
entails distances which could be used to determine different things such as
braking time for the car that's needed in different terrains, conditions, and 
speeds.

### Object Tracking
This ties into the last one with locality, but being able to track objects and how
they are moving from the systems view so that objects of similar type
don't become entangled, making a more cohesive system

### Pixel Identification
Being able to use semantic segmentation by identifying pixels available on the
screen and confidently saying what that pixel belongs to in relation
to the objects that appear on the road such as street, signs, pedestrians,
etc. We should be able to know what a pixel belongs to.

### Object Recognition and Classification
Being able to tell what objects are and potentially using techniques that
are able to identify specific types of objects such as types of cars, types
of signs, etc. These are all important factors when driving on the road,
the system being able to discern what an object is as if we were identifying
it could play key roles in how it evaluates situations.

### Driver Assistance
These ties in everything previously into this, but using the collected data
and the models evaluation, it's able to make decisions (a.k.a. suggestions)
like braking when there's a stopped car or red light, speeding up when the flow
of traffic is picking up, swerving when there's not enough stopping distance to brake
fast enough for an object in front of you, and more.

### Datasets
There is data posted on the canvas for this project, and I think it should be sufficient 
enough to choose from. In addition, there should be a validation set to see how the model
reacts to new data and determine whether there should be adjustments in order to avoid
overfitting and underfitting for specific data.I just need a certain amount of 
data to train on, then I need data to demonstrate the model for testing that's separate 
from its training data. The initial dataset should include a wide variety of roads 
and conditions for the driver so that there is more to look at for the model. 

### Techniques for object detection
We've already gone through some basics of object detection such as tracking by
hue/color. In addition to this, we can detect objects through shape based detection
which could be really useful in tandem with color based to detect road signs and
road lanes. We can also use the key points of features such as corners, edges, unique
patterns, textures, etc that are specific to an object that will allow us to have a
better object tracking mechanism


## Part 2
### Source Dataset

I acquired various videos (about 50 of various lengths) of different size from 
one of the recommended raw data sources for road imagery: 
https://www.cvlibs.net/datasets/kitti/raw_data.php
I downloaded 50 so that I could use 30 videos for training, 10 videos for testing, 
and 10 videos for the final demonstration. 

Since driving is a very complex task with all the different objects coming into
view from multiple angles, I don't think there needs to be a massive difference
between the training and validation besides just being different videos, 
as different videos imply different sceneries, people, objects, streets, 
etc. But, both the training dataset and validation, I through in a variety
between open road, residential, and city driving to see if the areas
in which it can operate can be validated.

Each video contains a minimum of 3 different types of objects which include, the road, 
cars, and signs. The road is obviously very braod and there's a lot to look at for the road,
but I'll count it as one object for simplicity sake. The number of these objects appearing 
varies based on the video being shown. Some videos (espcially of city driving) also feature 
pedestrians. Each video is just a bunch of images with a resolution of 1392x512 and has 
daylight as its lighting, although the datasets also include similar videos with different
lighting, such as greyscale. The weather is all daylight, and the camera recording are all 
dashcams which either have CMOS or CCD sensors.

## Part 3
### preprocessing

The data I have acquired is in frames rather than actual mp4 files. However, I want to use mp4
files as there's a library that can overlay an mp4 with whatever I need to add for
each frame, and on top of this, I took a suggestion from the previous part and recently got a dashcam.
I will be also collecting my own data here pretty soon, but for now, everything I have done
has been on the collected open source data. 

To accommodate for the fact that I have frames rather than mp4 files for dashcam footage, I created the 
mp4 files using cv2 in my create_video.py file. I'll give instructions in how to call create_video and 
extraction.py later on. But what this essentially does it it stitches frames in the correct ordering into
a videowriter object which then allows me to save the final video as an mp4. This is our input, then in order
to perform feature extraction we run extraction.py

for an example of how to run everything:

```sh
python create_video.py ./data/train/vid3/
python extraction.py
```

Note: using this example, going to about 1 minute into the video I think demonstrates the lane tracing the best

Then you'll have a video appear called output.mp4 that shows the features that have been attempted to be
extracted. As of right now, I'm able to detect the road lanes with an okay accuracy, but when the road is cluttered,
or the surroundings are crowded (and especially when there are train tracks), it becomes hard for
my algorithm to find the road. I'm also trying to extract features from cars, people, and road signs, and right now
I'm in the beginning stages of that by trying to extract features of contours I find based off of edges
I find using canny edge detections.

For feature extraction: What I am using to try to extract lines on the road is canny edge detection and 
using that to then find a specific region of where I'm extracting these lines from, then applying a hough
transform to show the lanes on the road (or at least what my program believes to be the lanes on the road).
For detecting shapes and objects that appear on the road I'm still in beginning stages and have only made efforts
to find objects based on area it takes up on the images. I want to extend this capabilities very soon and 
look at shapes and colors as a means for identifying objects as well. So far, it uses canny edge detection as well
since most objects on the road have defines edges, so canny edge detection is able to find where the edges in
the image are, then I want to use these edges to identify shapes that I can use to then classify these objects
into where they belong.

I put some examples of how my program segments the data under a folder called segmentations. There you 
will see a few images of how my program dissects the images to find what it needs to.