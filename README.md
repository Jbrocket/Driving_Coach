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

## Part 4
### Classifiers

##### Datasets
cars: https://www.kaggle.com/datasets/sshikamaru/car-object-detection/
roadsigns: https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/

As of right now, I've only classified one object (cars) using a deep learning algorithm that bases itself off of
Convolutional Neural Networks, YOLO. The reason for the choice in algorithm was the idea that YOLO reduces the chances
of false positives when classifying objects due to reducing class probability if there is another possible object 
in the frame, which lowers the chance of false positives. However, as you will see when analyzing the outputs of this model,
we see that not all cars are taken into account by the algorithm. Another benefit of this model is it will aim to predict
the bounding boxes of the object classifiers, meaning it will try to find the bounding box given an image, rather than needing to 
use a sliding window technique to pan the screen and find objects that way. As of right now, I'm also working on a YOLO 
model for road sign detection, and I will be including that into the project for the final part once it's completed.

Since this wasn't really in the cards when I planned this projects (I honestly didn't even know what it took to create
a roadside detector using neural networks), I added datasets that were used for classifying objects, and I also created
separate Kaggle workbooks since my local machine wasn't liking torch.
Here is the link for my car object detection model using YOLO:
https://www.kaggle.com/code/jonathanbrockett/yolo-car-object-detection/edit 

If you want to actually create your own model and run it, you will need to do so on that notebook. It's really easy to 
create a copy of mine and just run all.

Before creating this model, I experimented with a different model that you can see in src/car_detection/h5.py
This creates a simple neural network that was able to find singular objects (cars) with apporximately 80% accuracy,
but this model wasn't working too well once it came to the actual video because the train set is just very different
from the actual video in terms of quality and resemblance between the data. Especially since I collected my own
footage for this part of the project, the quality I was able to get from recording on the road was good, but lighting
differed greatly which I believe affected the model a lot.

Currently, my YOLO model predicts cars on the road with a Mean Average Point Precision of about 97.5% accuracy on 
the validation data, an F1 score at 0.3768, a Recall score of about 98%, and a Precision score of 0.2331.
well with my videos even though there is a discrepency, as I mentioned before, between lighting and quality of the actual dataset
and my recorded videos. Even though the lighting can be exceptionally bad, the model still is able to accurately find some 
cars within the vicinity with little false positives.

To view the before videos simply run:
```sh
python create_video data/videos/<chosen_video>
```
and the output will be under output.mp4. Don't run part3 in the way I described earlier in the semester as it's now deprecated with
all the changes that have been added since it won't actually use the YOLO model.

and to view the videos that were made using the YOLO model and my preprocessing to find road signs run:
```sh
python create_video data/processed_videos/<chosen_video>
```

What was achieved:


extractionv2.py is what I used in my kaggle and it has an enhanced algorithm for finding lines on the road. Other than that, I found
a dataset related to car detection on the road, and then I used that dataset to train two models: the h5.py model which is a simple network 
with lots of parameters (you can look at the file), and I trained a YOLO model for the same dataset that outperforms the h5.py model. This 
took a considerable amount of time especially since I've been working alone, and I hope to get sign detection up and running before 
the final turn in data for the final report. There is an example run of my kaggle notebook in this google drive (it was too big of a file):
https://drive.google.com/drive/folders/1GA_3-uQQk7I1jObTi7rDkwR8mAHRqAxd?usp=sharing 
This also contains a model that was too big to upload as well.

EDIT: The model for h5 is in models/ and the notebook is in src/ so you can just look for these files in the repository


## Part 5 FINAL

### Dataset
https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data
https://www.kaggle.com/code/jonathanbrockett/yolo-car-object-detection/edit ### MY CODE TO TRAIN IT
I was only able to implement car detection by trianing a yolo-nas model on this dataset.
1001 training images and 175 testing images.
The results are as follows: 
Mean Average Point Precision: 0.975
F1: 0.3768
Recall: 0.08 
Precision: 0.2331.

So while the model is good at creating the bounding boxes, it can be prone to making false positives at times and also
doesn't always spot the cars. This is actually evident when watching the processed videos I created where Cars do get
picked up by the model, but it's inconsistent mainly based on lighting and view of the car given. For instance, it's more
likely to pick up the side of a car rather than the front, and this might be due to the fact that the dataset is mainly
comprised of side views. What is interesting though is since the bounding boxes are really good, you can make an annotated
dataset based on this model alone by taking anything with a high confidence and saving the frame and bounding box. These were
the measurements achieved on the test data, but the test data is completely separate from the application. If we wanted to see
better results on the test data, then I think we would need a bigger annotated dataset to train on and more epochs. The dataset
itself was only about 1001 annotated pictures and the variety collected didn't vary much which is what I believe led to issues. 
Having more data should help the model train better on certain profiles of a car such as the front and backsides which 
it doesn't seem to pick up as easily. On top of that, I think if that were the case it would also perform better on the 
processing of actual videos as well since one of its major weakpoints was dealing with certain angles of a car that it probably 
wasn't as used to. A variety of lighting could've helped as well (more data is the theme here) since the annotated dataset
I found mainly comprises of very similar angles and lighting of cars on a specific type of road. So a bigger variety in 
environemnt, lighting, car angles, etc would've been better in both creating a model that performs in test and applying it
to actual roadside footage. The model overall does a pretty decent job at spotting cars, it's noticeably better in certain
circumstances where the lighting is good and the car sides are more apparent, but it's a good starting place in being
able to identify objects that are on the road. 

In terms of contribution, since this is a solo project I did everything including training the model, creating the hough
transform to detect lane lines, processing the videos through the different modifications that were suggested by the model
and by the hough transform, finding the dataset to use, and some data collection using a dash cam as well for actual
videos to demonstrate the product on.

To see how I used the mdoel and line detection in code, look at extractionv2.py which has a lot of my implementation work,
then look at https://www.kaggle.com/code/jonathanbrockett/yolo-car-object-detection/edit and just run all.

To Create a video that I already processed frame by frame with the car-detection model and my lane detector run
```sh
python create_video data/processed_videos/<chosen_video>
```

There should be two output files already in the directory which is what you get from running the above command.