##Vehicle Detection Project

[//]: # (Image References)
[img_car_not_car]: examples/car_not_car.png
[ycrcb]: examples/ycrcb.png.jpg
[img_features]: examples/img_features.png
[sliding_windows]: examples/sliding_windows.png
[detected_boxes]: examples/detected_boxes.png
[heatmapcontours]: /examples/heatmapcontours.png
[falsepositive]: /examples/falsepositive.png

[video]: project-video-output-with-lanes.mp4

---

Steps of the project:

1. Train a classifier to recognize vehicles
    - Convert the images to YCrCb space
    - Combine features from histogram of oriented gradients on all 3 channels and color histogram and normalize them
    - Train a Linear SVM after normalizing features and creating a random test set

2. Recognize vehicles on a single image
    - Implement a sliding-window method
    - Run the classifier on each window

3. Recognize vehicles on a video:
    - Combine the vehicles identified in consecutive frames to reject false positives and refine the bounding box around the vehicle
    - Overlay lane detection from previous project

---

#1. Train a classifier to recognize vehicles

The code is in the `vehicle-detection-train-svm.ipynb` notebook using functions from `vehicledetectiontrainsvm.py`.
`vehicledetectiontrainsvm.py` contains functions that are used to train the classifier, and also by the video pipeline that uses the classifier to determine the location of the vehicles.

## Data
I used the labeled dataset provided by Udacity. It contains  images of [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles] (https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).

![img_car_not_car][img_car_not_car]


There are a total of 18,458. 48% of them are cars and 52% of them are non-cars, so the classes are almost balanced. 
The size of the images is (64, 64, 3).

I used `scipy misc.imread()` to load .jpg and .png images consistently.

## Extract Image features

The images are converted to YCrCb. This was the color space that achieved the best results with the classifier. 
Y represents the luminance, while Cr and Cb are the red-difference and blue-difference chroma components (see [link] (https://en.wikipedia.org/wiki/YCbCr)).
Other color spaces I tried include HSV, LUV, HLS and YUV. 

![ycrcb][ycrcb]

The function `extract_features()` in `vehicledetectiontrainsvm.py` concatenates features from Histogram of Oriented Gradients (HOG) and Histogram of color.

### Histogram of Oriented Gradients (HOG)
The `get_hog_features()` function in `vehicledetectiontrainsvm.py` returns HOG features for a given channel of the image.
We are running it for all 3 channels of the image with the following parameters: orientations = 8, pix per cell = (8, 8), cell per block = (2, 2).

4,704 features are created (1,568 per channel).

To fine-tune the parameters, I plotted what they looked like for a few images of vehicles and non-vehicles. I also ran the model with different parameters and picked the ones with the best accuracy.

### Histogram of color
The `color_hist()` function in vehicledetectiontrainsvm.py returns a histogram of color for all 3 channels with a bin length of 32.
96 features are created (32 per channel).

![img_features][img_features]

The total number of features created is 4,800.
I've tried to add spatial features but they did not improve the accuracy of the model.

## Normalization
This step is done in `vehicle-detection-train-svm.ipynb` (cell [8]).
I used the sklearn.preprocessing `StandardScaler()` function to normalize the features values. This is very important since the HOG features are between 0 and 0.31 and the histogram of color features are between 0 and 3,605. If we did not normalize, features from the histogram of color would be given more weight in the model since their values are much higher.

## Training/Test set
This step is done in `vehicle-detection-train-svm.ipynb` (cell [8]).
I used the sklearn.model_selection `train_test_split()` function to randomly select 20% of the data for the test set.

## Train model
This step is done in `vehicle-detection-train-svm.ipynb` (cell [9]).
I used a SVM model with a linear kernel `sklearn.svm.LinearSVC()` and parameter C = 1. I picked SVM because they tend to perform well in high dimensional spaces.
To get access to probability estimates, I used `sklearn.calibration.CalibratedClassifierCV()`. 

The model took 26.7 seconds to train, and had a training accuracy of 1.0 and a test accuracy of 0.99. 
HOG features alone achieved a test accuracy of .98.

To predict a single example, it takes 0.001 seconds. It's important for the predictions to be fast since we'll run 831 predictions per frame (see below for calculation), and the project video has over 1,260 frames!

## Save model
I saved the trained model in `finalized_model.sav` and the scaler used in `finalized_scaler.sav` so that they can be reloaded in other notebooks.

# 2. Detect vehicles in a single image
The code for this part is in the `vehicle-detection.ipynb` notebook.

### Sliding Windows 

I created two sets of overlapping windows:

- A big window of size 128x128 with 80% overlap on both dimensions. As cars won't appear in the sky, I'm restricting the windows to be from y = 380 to the bottom of the window.
- A small window of size 64x64 with 80% overlap on both dimensions. As smaller cars will be around the horizon, I'm restricting the small windows to be from y = 400 to y = 500.

I originally also had an even bigger window (196x196) but I got rid of it.
As the car can be anywhere along the horizontal axis, windows cover the entire horizontal axis.

There are 831 windows: 423 big windows and 408 small windows.

I tried different values for the overlapping percentage, ranges of windows and sizes of windows. I tested them on selected images first, and then on the video, before finalizing them.

![sliding_windows][sliding_windows]

### Predict where the vehicles are in the image

The `pipeline_frame()` function calls `get_window_recognition()` for the 2 sets of windows defined above.

`get_window_recognition()` first calls `slide_window()` to create all the windows to search. 

It then runs the classifier on each window:

- It resizes the window image to 64x64 since that's the size of the images used to train the classifier.
- It calculates the features of image (HOG and Histogram of color) using the same function as the one used to train the classifier.
- It normalizes the features using the same scaler as the one used to train the classifier.

It returns:

- The list of windows where a vehicle is detected.
- The list of probabilities that there is a vehicle in the detected windows.

![detected_boxes][detected_boxes]

---

#3. Detect vehicles in a video

## Combine frames to create a heatmap and find contours

The `pipeline_combined()` function keeps the detected windows and their associated probabilities of 2 consecutive frames.
It then calls the `get_heatmap()` and `find_contours()` functions defined below.

### Create a heatmap
This is done by the `get_heatmap()` function.

- It creates a heatmap by summing the probability of each pixel to contain a car from 2 consecutive frames. We used 2 consecutive frames to get more robust data. Using more frames would have a detrimental impact since the false positive detections will remain for a longer time.
- It applies Gaussian blur using the `cv2.GaussianBlur()` function.
- It uses the `cv2.threshold()` function to create a binary threshold from the heatmap. It only keeps the pixels whose value is above 1.4. This threshold value was chosen after testing extensively on the video. Pixels can be part of multiple overlapping window so some pixels have values above 8. The minimum possible value is 0.5 (pixel detected only in 1 frame, by a single window of probability 0.5).

It returns the thresholded binary heatmap, as well as the heatmap values that will be used later.

### Find contours
This is done by the `find_contours()` function.

- From the thresholded binary heatmap, it uses the `cv2.findContours()` function to identify the contours.
- Using the heatmap values, it calculates the maximum heatmap value of each contour. This will be helpful later to know how confident we are that there is a car in a contour.

![heatmapcontours][heatmapcontours]

## Define a Car class
When we believe that a car is contained in a frame, we create a car object, which has the following attributes:

- cxs / cys: 2 lists containing the last 10 x / y values of the centroids of the car
- xs / ys / ws / hs: 4 lists containing information about the rectangle surrounding the car (top-left corner x value, y value, width of rectangle, height of rectangle)
- cx / cy: average of the last 10 recorded x / y values of the centroids of the car
- x / y / w / h: average of the last 10 recorded x / y values of the top-left corner of the surrounding rectangle, its width and its height
- detected: was the car detected in the latest frame
- undetected_count: for how many frames was the car undectected
- detected\_from_match: for how many frames was the car detected from a match (explained later)
- detected\_count_needed: for how many frames in a row should the car be detected before we are confident enough that it is a car and display it. The default is 10.
- images: list of the last 10 images of the car

## Define a Cars class
- Contains the list, named cars, of detected cars.

## Video pipeline
The logic below was implemented to minimize the false positives. It can be found in the `pipeline_video()` function.

Example of false positive:

![falsepositive][falsepositive]

The general logic is:

When a contour is detected, we assume that it corresponds to a car and we try to associate it to an existing car, or create a new car.
Before it is displayed, a car must be detected for a certain number of frames in a row. This number of frames depends on the maximum heatmap value of the contour. The higher it is, the more confident we are that it is a vehicle and the fewer frames we need.

In more details:

For each frame, the `pipeline_combined()` function described above returns the list of contours of the image and their maximum heatmap value.

For each contour, we use the `cv2.boundingRect()` function to get the coordinates of the top-left corner of the surrounding rectangle, its width and its height (x, y, w, h). The `cv2.moments()` function gives the coordinates of the centroid (cx, cy).

If the width of the height of the rectangle is below 50px, it is ignored.

We first go through the detected cars to see how many of them are contained in the contour.

- More than 1 car: 

 Double check that the cars are really in the frame:
We use the `cv2.matchTemplate()` function to look for the latest car image (from car.images) in the frame. If the car is found and the distance between the previous centroid of the car and matched location is less than 50px, we are confident that the car is indeed in the contour.
	- Do the multiple cars contained in the contour overlap? Under the assumption that the top-left corner of the car is now located where the match was found and that the width and height of the car are the same as before, we calculate the overlap between the rectangle of the cars contained in the contour. If they overlap by more than 40px, we consider that there is an overlap.
		- If overlap: we delete the existing cars and create a new one using the new contour. Because it comes from existing cars, we are confident that it is indeed a car and force the detected_count to be equal to 10, so that it is displayed right away. 
		- If no overlap: the cars are most likely distinct, the find_contour() function may have grouped them by mistake. we update the centroids of all the cars contained in the contour using the result of `cv2.matchTemplate()`.
- 1 car or less:
	- If a car is contained in the contour:
		- If the maximum heatmap value of the contour is greater than 4, we are quite confident that the contour represents a car and we can update the detected_count_needed of the car to 3. This means that the car only needs to be detected in 3 frames in a row before being displayed (instead of 10 by default)
		- If the contour is big enough (over 200px wide or more than 2/3 of the current width of the car), the detection is fine and we can safely add the new contour information to the car object. (i.e. add contour information to list of centroids, rectangle and images of the car)
	- If there is no existing car in the contour:
		- If there is an existing car within 50px of the contour
			- If the car is close to no other contour and the contour has a good size (at least 2/3 of the current width of the car or more than 200px), we add the contour information to the car.
			- If the car is close to another contour
				- If the combined contours are about the size of the car and are not too wide (under 250px), they are probability both part of the existing car and we can assign the combined contours to the car. This can happen if the vehicle is large and non-overlapping windows identfty different parts of the same vehicle.
				- If the combined contours are about the size of the car but are too wide, we delete the car and create 2 new ones. Because the combined contours come from an existing car, they are probably cars as well and I force the detected_count to 10 so that the car will be displayed right away. This can happen if a car is hidden by another one and slowly appears.
			
- If there is no close car:
We create a new car for this contour. If the maximum heatmap value of the contour is greater than 4, we can be quite confident that it is a car and we'll require only 3 consecutive detections of the car before displaying it (car.detected\_count_needed = 3). Otherwise, we'll wait for 10 consecutive detections.

Finally, we go through the cars that were previously displayed, which means that we were very confident that they were cars, but not assigned to any contour in this frame.

- If the car's centroid is now part of another car's rectangle, we update the detected_count attribute of the other car to 10 (since it's replacing a car we were confident about) and delete the previous car.
- Using the saved images of the car (stored in car.images), we check if the car can be found within 50px of its previous rectangle. We use the `cv2.matchTemplate()` function for that task. If so, we'll consider the car detected in the frame. However, we'll increment its detected\_from_match attribute to remember that it was not detected from the contour.

To be displayed, a car must be detected in the current frame and have been detected for enough frames in a row. The number of frames depends on the maximum heatmap value of the contour. If the contour had a maximum heatmap value of at least 4, we are confident enough that a car is there and we only require 3 consecutive detections. Otherwise, we require the car to be detected on at leat 10 contours in a row.

Cars can also be deleted if:

- Car had been detected less than 7 times so far and was not detected in the latest frame
- The last 12 times a car was detected, it was from an image match
- A car is not detected at all for 5 frames in a row

## Overlay detected lanes:
We import the function to return the image with the detected lanes from the previous project: `from P4AdvancedLaneFinding import full_pipeline as full_lane_pipeline`

The video pipeline calls the `full_lane_pipeline()` function that returns an undistorted image with the lanes.
It uses the distorted image to detect the vehicles and draws the rectangles on the undistorted image that contains the lanes.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took is very sensitive to the parameters such as window sizes, overlap, heatmap binary threshold, number of frames to consider in the heatmap. I picked them so that there is a limited number of false-positives. My approach works well on the project video but, when I tried to run it on undistorted images (so that I could overlay the lanes), some false-positives appeared.

Instead of using `cv2.findContours()`, I had originally tried to detect blobs using the Determinant of Hessian method from `skimage.feature.blob_doh`, followed by watershed segmentation to identify the pixels that belong to each blob using `skimage.morphology.watershed`. I managed to make this approach work on a few images, but, when testing on the video, it was failing on too many frames. `skimage.feature.blob_doh` was too sensitive to the parameters and I was never able to find parameters that work on all the frames.

To improve the detection and therefore reduce the false positives, we could feed more images to the Linear SVM.

The next step I'm most excited about is to use a deep learning approach to detect the cars. 

Finally, the pipeline is currently too slow to be used real time.


