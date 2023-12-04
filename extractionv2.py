import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import cv2
from moviepy import editor
import moviepy
import tensorflow as tf

import math
from statistics import mean

car_detection_model = tf.keras.models.load_model("models/car-object-detection.h5")

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # IMPORTANT: We will use an alternative grayscale method that enhances the yellow component
    # by eliminitating the Blue component.
    r,g,b = cv2.split(img)
    grayed_img = np.uint8(.5*r+.5*g)
    return grayed_img
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    #for line in lines:
    #    for x1,y1,x2,y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    color2 = [0,255,0]  # auxiliar green color
    centers = [[],[]]  # Array of two arrays, one for each side.
    w_slope = [0,0] # weighted slope, two values: left and right
    total_l = [0,0] # total length, two values: left and right
  
    for line in lines:
        for x1,y1,x2,y2 in line:
            s= (y2-y1)/(x2-x1)      # s is the slope of the line
            if (0.4 < abs(s) < 1):  # discard outliers
                c_xy= [.5*(x1+x2), .5*(y1+y2)]    # central point of the line
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if(y2<y1):  # left side case
                    if(x2>x1) : # discard outliers like line pointing to the left on the left side
                        w_slope[0] += s * length  #slope weighted by length of line
                        total_l[0] += length
                        centers[0] += c_xy
                        #cv2.line(img, (x1, y1), (x2, y2), color2, 2)
                
                if(y2>y1):  # right side case
                    if(x2>x1): # discard outliers like line pointing to the right on the right side
                        w_slope[1] += s * length  #slope weighted by length of line
                        total_l[1] += length
                        centers[1] += c_xy
                        #cv2.line(img, (x1, y1), (x2, y2), color2, 2)
    
    top_y = int (img.shape[0] * .62)
    
    for i in range(2):
        if(total_l[i]>0):
            avg_slope = w_slope[i]/total_l[i]
            avg_x = int(mean(centers[i][::2]))
            avg_y = int(mean(centers[i][1::2]))
            x1 = avg_x + int ( (img.shape[0] - avg_y)  / avg_slope )
            y1 = img.shape[0]
            
            y2 = top_y
            x2 = avg_x + int ((y2-avg_y)/avg_slope )
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        else:
            print("Cannot draw slope line !")
    
    return
        

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
    Parameters:
        image: image of a road where one wants to detect lane lines
        (we will be passing frames of video to this function)
    """
    preprocessed_frame = cv2.resize(image, (64, 64)) / 255.0  # Resize to match the model's input shape

    # Make predictions using the model
    predictions = car_detection_model.predict(np.expand_dims(preprocessed_frame, axis=0))
    
    # Grayscale
    grayed_img = grayscale(image)
    
    #Before detecting the edges, we should apply a gaussian filter to soften noise
    gauss_img = gaussian_blur(grayed_img, 5)
    
    #Now we apply the Canny Edge detector on the grayed image:
    canny_edges_img = canny(gauss_img, 40, 80)
    
    #Now we apply a region mask to focus only in our region of interest.
    #The region of interest will be defined by a trapezoid with two vertex in the bottom corners
    #of the image and the other two in the 'horizon' area of the road aprox.
    dims= canny_edges_img.shape
    top_vertex_l = (dims[1]*.45,dims[0]*.62)
    top_vertex_r = (dims[1]*.55,dims[0]*.62)
    vertices = np.array([[(0,dims[0]),top_vertex_l,top_vertex_r,(dims[1],dims[0])]], dtype =np.int32)
    masked_img = region_of_interest(canny_edges_img, vertices)
    
    #Finally we apply the Hough transform to detect the lines in the image.
    lines_img = hough_lines(masked_img,1,np.pi/180,40,30,200)
    
   
    result = weighted_img(lines_img,image)

    # Process the predictions and draw bounding boxes
    pred_bbox = predictions[0]

    # Map predicted bounding box coordinates back to the original resolution
    original_shape = result.shape[:2]
    pred_bbox_original_resolution = [
        pred_bbox[0],  # xmin
        pred_bbox[1],  # ymin
        pred_bbox[2],  # xmax
        pred_bbox[3]  # ymax
    ]

    # Draw bounding box on the original frame
    xmin, ymin, xmax, ymax = map(int, pred_bbox_original_resolution)
    cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return result
 
# driver function
def process_video(test_video, output_video):
    """
    Read input video stream and produce a video file with detected lane lines.
    Parameters:
        test_video: location of input video file
        output_video: location where output video file is to be saved
    """
    input_video = editor.VideoFileClip(test_video, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, audio=False)
    
if __name__ == "__main__":
    import sys
    process_video(sys.argv[1],'output.mp4')