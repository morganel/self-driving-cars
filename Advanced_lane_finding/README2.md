##Advanced Lane Finding Project
---

****

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[camera-cal-img-test-image]: /camera_cal/calibrationtest.jpg "Calibration Test Image"
[camera-cal-img-undistorted-image]: /camera_cal/test_undist.jpg "Undistorted Image"

[Distortion_correction_original_image]:test_images/solidWhiteRight.jpg "Distortion_correction_original_image"
[Distortion_correction_corrected_image]:output_images/undist_solidWhiteRight.jpg "Distortion_correction_corrected_image"

[birdeye-img-original]: output_images/birdeye_original.jpg "Original image"
[birdeye-img-warp]: output_images/birdeye_warp.jpg "Birdeye image"
[birdeye-img-unwarp]: output_images/birdeye_unwarp.jpg "Unwarped image"

[masks-original]: output_images/masks_pipeline_original_img.jpg "Original image"
[masks_yellow_white]: output_images/masks_yellow_white.jpg "Yellow and white binary"
[masks_L_Dir_Magn]: output_images/masks_L_Dir_Magn.jpg "L - gradient direction and magnitude binary"
[masks_L_Dir_Magn_V]: output_images/masks_L_Dir_Magn_V.jpg "L - gradient direction and magnitude + V binary"
[masks_pipeline_result]: output_images/masks_pipeline_result.jpg "Final binary"

[identify-lane-image]: output_images/identify_lane_orgininal_mask.jpg "Original image"
[identify-lane-histogram]: output_images/identify_lane_histogram.jpg "Histogram bottom half image"
[identify-lane-slices]: output_images/identify_lane_slices.jpg "Histogram for each slice of the image"
[identify-lane-both-lanes-pixels]: output_images/identify_lanes_both_lanes_pixels.jpg "Pixels of identified lanes for left and right"

[identify-lane-v2-image]: output_images/identify_lane_v2_birdeyeimage.jpg "Original image"
[identify-lane-v2-mask]: output_images/identify_lane_v2_existing_mask.jpg "Mask from existing lanes"
[identify-lane-v2-lane]: output_images/identify_lane_v2_identified_lane.jpg "Identified lane"

[fit-poly-fill]: output_images/fit_poly.jpg "Fill polynomial"
[fit-poly-original-space]: output_images/fit_poly_original_space.jpg "Lanes marked"

[video1]: ./project_video.mp4 "Fit Visual"

---

###Camera Calibration

The code for this step is contained in the IPython notebook `camera_calibration.ipynb`.

I isolated one image `camera_cal/calibration1.jpg` to use as test image and renamed it `camera_cal/calibrationtest.jpg`.

####Compute the camera matrix and distortion coefficients
I used the remaining 19 images in the `camera_cal` folder to calculate the camera matrix and distortion coefficients.

Looking at the images, there are 9x6 inner corners in the chessboards.

Step 1:
`objpoints` will store the list of 3d points in real world space (we'll assume z=0) and `imgpoints` will store the 2d points in image plane.
For each image:

- Convert to gray scale
- Identify corners using `cv2.findChessboardCorners`
- If all corners are successfully identified, append the same array of coordinates `objp` to `objpoints` and append the detected corners to `imgpoints`.

All corners were successfully detected in 16 out of 19 images. 

Step 2:
Feed `objpoints` and `imgpoints` to the function `cv2.calibrateCamera()` that returns the camera matrix and distortion coefficients.

#### Test camera calibration on test image
I tested the distortion coefficients and camera matrix on test image using `cv2.undistort()`. 

![Test image][camera-cal-img-test-image]
![Undistorted Image][camera-cal-img-undistorted-image]

###Pipeline (single images)

Code is in IPython notebook called `P4-Advanced-Lane-Finding.ipynb`.

#### Step 1. Distortion correction
I used the camera matrix and distortion coefficients obtained above to correct all images.

Example:
Original image                                           |  Corrected image
:-------------------------------------------------------:|:-------------------------------------------------------:
![original image][Distortion_correction_original_image]  |  ![corrected image][Distortion_correction_corrected_image]

#### Step 2. Perspective transform to rectify the image and create bird-eye view.

I used `cv2.getPerspectiveTransform()`.  The `cv2.getPerspectiveTransform()` function takes as inputs source (`vertices`) and destination (`dst`) points.  Below are the values of `vertices` and `dst`.

``` 

pt1 = (50,img_size[0])
pt2 = (int(img_size[1] * .4), int(img_size[0]/1.5))
pt3 = (int(img_size[1] * .6), int(img_size[0]/1.5))
pt4 = (int(img_size[1]) - 50, int(img_size[0]))
vertices = np.expand_dims(np.float32([pt1,pt2,pt3,pt4]), axis=0) 

dst = np.float32( [[0,img_size[0]], [0,0], [img_size[1],0], [img_size[1],img_size[0]]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 50, 720      | 0, 720        | 
| 512, 480      | 0, 0      |
| 768, 480     | 1280, 0     |
| 1230, 720      | 1280, 720        |

I verified that my perspective transform was working as expected by drawing the `vertices` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. I also made sure that I could unwarp the image.

Original image|  Warped image| Unwarped image| 
:---------------------------:|:---------------------------:|:---------------------------:
![original image][birdeye-img-original]  |  ![warped image][birdeye-img-warped] |  ![unwarped image][birdeye-img-unwarped]

#### Step 3. Create binary image by using color transforms and gradients
This resulted in the function `pipeline()` in the 7th cell of the notebook.

- Yellow binary:
Convert image to HSV and select pixels whose value is between 
`lower_yellow  = np.array([ 0,  100,  100])` and 
`upper_yellow = np.array([ 80, 255, 255])` using `cv2.inRange`. 
These values were obtained empirically after testing on images.

- White mask:
Keep image in RGB and select pixels whose values are between
`lower_white  = np.array([ 200,  200,  200])` and
`upper_white = np.array([ 255,  255, 255])`.

Example:
![Yellow and white binary][masks_yellow_white]
![Original image][masks-original]

These 2 binaries were good enough to pass the "Project video". However, it was failing on the "Challenge video". So I added more binaries:

- L channel from HLS: binary using gradient magnitude and direction AND the V channel from HSV threshold
-- Convert image to HLS space. The L channel seems to identify the lanes properly and does not contain as much noise as the S channel.
Using a sobel kernel size of 15, create a binary image where the gradient magnitude is within `(30, 255)` and the gradient direction is within `(0.1,0.8)`.
This identified lanes properly but also identified vertical changes of color of the road due to road work etc.

Example:
![L - gradient magnitude and direction binary][masks_L_Dir_Magn]

-- Therefore, I added another mask on top of it. I forced the previous binary to also have high values of V channel from HSV in `(180,  255)`.
That eliminated the lanes of darker colors that were not really lanes.

Example:
![L - gradient magnitude and direction binary AND V threshold][masks_L_Dir_Magn_V]

- Union the 3 previously defined masks
Select pixels that belong to 'White mask' or 'Yellow mask' or 'L channel from HLS: mask using gradient magnitude and direction + V channel from HSV threshold'

Example:
![final binary][masks_pipeline_result]

####4. Identify the lane pixels (when we have no idea where the lane is)

`identify_lane` function.

Step 1: 
Run histogram search on the bottom half of the image to identify the peaks in intensity. 

![Original image][identify-lane-image]
![Histogram bottom half][identify-lane-histogram]

Identify the highest peak for each lane `argmax_histogram_side`: For left lane, look in the left half of the image and for the right lane, look for the peak in the right half of the image.
If nothing is detected, do a broader search (100-500 for left lane and 800-1200 for right lane)

Step2: Define mask

- Divide the image vertically in 8 slices. 

- side_ranges will record the x-range of the location of the lane for each of the 8 slices.

- Start the search in the  bottom slice of the image for x-values ranging from `argmax_histogram_side - 50` to `argmax_histogram_side + 50`. Find the x value `avgpoint` that has the highest intensity in that slice. Append `avgpoint - 50` and `avgpoint + 50` to side_ranges. 

- Move to the next slice (up) and find the new x value `avgpoint` that has the highest intensity in that slice (for x-values between `previousAvgpoint - 50` and `previousAvgpoint + 50`). 

- Iterate over the 8 slices.

Example for right lane:
![Lane slices][identify-lane-slices]

Step3:
Extract pixels of the image that are in the mask defined above. These pixels are the lane pixels.
If there are less than `_MIN_PIXELS_ = 500` pixels in a lane, discard it, since it's not enough to define the lane.
![Lane pixels][identify-lane-both-lanes-pixels]

#### 5. Fit lanes with a polynomial
Use `np.polyfit()` to fit a second order polynomial to the lane pixels identified above.

#### 6. Plot fitted lane back to original space
1. `draw_lanes_image()` function uses `cv2.fillPoly()` to fill in the region between the lanes defined by the polynomial.
![Draw region][fit-poly-fill]

2. Unwarp the figure and add it to the original undistorted image
![Lanes marked on image][fit-poly-original-space]

#### 7. Estimate radius of curvature of each lane and the position of the vehicle with respect to center in the lane

1. To estimate the radius of curvature of each lane in meters, we first have to convert pixels to meters. I estimated the following conversion parameters by looking at the birdview images:

`ym_per_pix = 12/720 meters per pixel in y dimension`
`xm_per_pix = 3.7/1000 meters per pixel in x dimension`

2. Refit a second order polynomial `side_fit_cr` to the lane pixels converted to meters using `np.polyfit()`

3. `get_curb()`  uses the formula below to calculate the radius of curvature at `y = y_eval` for the 2nd order polynomial `side_fit_cr`:
`side_curverad_meters = ((1 + (2*side_fit_cr[0]*y_eval + side_fit_cr[1])**2)**1.5) /np.absolute(2*side_fit_cr[0])`

I evaluated it at y_eval in the middle of the image since the values appeared more stable.

4. `get_car_position()` calculates the absolute distance in meters between the center of the car and a given lane (left or right lane).

5. `get_car_from_middle()` evaluates the distance in meters between the center of the car and the middle of the lane.
`distance_left_lane_to_car` and `distance_right_lane_to_car` are calculated using the function `get_car_position()` above.
The `road_center` is at `(distance_left_lane_to_car + distance_right_lane_to_car)/2`. Therefore, the distance between the car and the center of the road is `road_center - distance_left_lane_to_car`.
It is positive if the car is closer to the right lane and negative if it is closer to the left.

---

###Pipeline (video)

#### 1. Adapt lane pixel detection

If we have detected a lane in the previous frames, we know where the lane should be. Therefore, instead of blindly screening the entire image to look for a lane, we can search around the existing lanes.

- Divide the image in 10 vertical slices.
- For each slice:
-- get the middle y value and use the previous fitted 2nd order polynomial to get the expected x-value of the lane.
-- Look for the lane in a window of +/- 25 pixels around that x-value.

This not only makes the search more efficient, but it also avoids outliers.

Example for right lane:
![original image][identify-lane-v2-image]
![mask from existing lanes][identify-lane-v2-mask]
![identified lane][identify-lane-v2-lane]

#### 2. Ignore a detected lane in a frame
There are situations where it is better to discard a detected lane if we believe it is not accurate. We will discard a lane in a frame if:
- the newly detected lane is too different from the existing ones i.e. the distance between the first fit coefficient and the current best fit coefficient is greater than 0.05%.
- the ratio of the calculated radius of curbature of the left and right lanes, or its inverse, is greater than `_MAX_RADIUS_RATIO_ = 10`

#### 3. Avoid jittering by averaging the coefficients over the last `_N_ = 10` lanes
- Keep track of the last 10 'non-ignored' lanes.
- Average the values of the coefficients of the fitted 2nd order polynomial for the last 10 lanes into `best_fit`.

#### 4. Re-evalutaion of the radius of curvature and position of the car
They are re-estimated using the averaged lanes in the function `get_best_curb()`:
- Get 10 y-values across the y-axis. 
- Use `best_fit` to calculate the associated x-axis values.
- Convert the x and y  values from pixels to meters. 
- Refit a 2nd order polynomial.
- Get the radius of curvature of the lane using the same formula as in the previous section.

#### 5. Add plots around video: 
To help understand what each step does, I added the following images in the video:
- Bottom 4 images:
-- "Yellow/White" = Binary using the yellow and white filters
-- "L magn/dir" = Binary from "L - gradient magnitude and direction"
-- "L magn/dir + V"= Binary from "L - gradient magnitude and direction binary AND V threshold"
-- "Binary Output" = Combination of "Yellow/White" and "L magn/dir + V" binaries

- Bottom right left corner: Identified lanes (left lane in green and right lane in red)
- Top right corner: original image and bird-view image.

#### 6. Pipeline works great on both project and challenge videos.
Here's a [link to my project video](project_video_lanes.mp4)
Here's a [link to my challenge video](challenge_video_lanes.mp4)