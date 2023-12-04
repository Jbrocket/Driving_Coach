import cv2
import os

DIR = "vid10"
try:
    os.mkdir(f"/Users/jonathanbrockett/Workspace/Driving_Coach/data/videos/{DIR}_frames")
except:
    pass
PATH = f"/Users/jonathanbrockett/Workspace/Driving_Coach/data/videos/{DIR}_frames"
vidcap = cv2.VideoCapture(f'/Users/jonathanbrockett/Workspace/Driving_Coach/data/videos/{DIR}/MOVI0007.mov')
success,image = vidcap.read()
count = 0
while success:
    image = cv2.resize(image, (676, 380)) 
    cv2.imwrite(f"{PATH}/frame%d.jpg" % count, image)     # save frame as JPEG file   
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1