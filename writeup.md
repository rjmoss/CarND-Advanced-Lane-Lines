## Rob Moss Advanced Lane Finding writeup

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/original_chessboard.jpg "Original Chessboard"
[image2]: ./output_images/undistorted_chessboard.jpg "Undistorted Chessboard"

[image3]: ./output_images/Original.jpg "Original"
[image4]: ./output_images/Undistorted.jpg "Undistorted"
[image5]: ./output_images/Threshold_(combined).jpg "Threshold"
[image6]: ./output_images/Mask.jpg "Mask"
[image7]: ./output_images/Warped_Threshold.jpg "Warped Threshold"
[image8]: ./output_images/Lane_Lines.jpg "Lane Lines"
[image9]: ./output_images/Lane_filled.jpg "Lane Filled"
[image10]: ./output_images/Lane_unwarped.jpg "Lane Unwarped"
[image11]: ./output_images/Result.jpg "Result"

[image12]: ./output_images/src_points.jpg "Source Points"
[image13]: ./output_images/dst_points.jpg "Destination Points"
[image14]: ./output_images/transform_unwarped.jpg "Warped Unwarped"

[image15]: ./output_images/histogram.jpg "Histogram"

[video1]: ./output_images/video_result.mp4 "Video"
[video2]: ./output_images/challenge_result.mp4 "Challenge"
[video3]: ./output_images/harder_challenge_result.mp4 "Harder Challenge"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

The code is all in a python notebook called `pipeline.ipynb` which was used for development, and the pure python script is in a normal python file called `pipeline.py`.

### Writeup / README

You're reading it...

### Camera Calibration
The camera is calibrated at the start of the python notebook or in lines XXX to XXX of `pipeline.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

*Original Chessboard*
![alt text][image1]

*Undistorted Chessboard*
![alt text][image2]

Note that the camera only needs to be calibrated once so the the resulting calibration is pickled into `camera_cal/pickled_calibration.p`.

### Pipeline (single images)

#### 1. Distortion Correction

The distortion is applied to the original image (see below). The change can be slight and tricky to spot but compare the sign post on the left between the images or the amount of car hood in the bottom right corner.

![alt text][image3]
![alt text][image4]

#### 2. Thresholded binary

Through a mixture of trial and error and careful tweaking I settled on a combination of an x-gradient threshold, lightness threshold and saturation threshold. The code can be seen in the `threshold_combination` function (line XXX of `pipeline.py` but is in summary:
```
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=KSIZE, thresh=(20, 100))
saturation_binary = hls_select(img, thresh=(180, 255))
lightness_binary = lightness_select(img, thresh=(230, 255))

combined = (gradx | saturation_binary | lightness_binary)
```
Applied to the test image above the lane lines have been isolated fairly well (though this is an easier example than some of the frames from the cideos - see below).

![alt text][image5]

##### Mask

In addition to the thresholding I also applied a mask to remove the parts of the image which we are not interested to remove distractions which could cause issues later in the pipeline. The following mask was applied to create a masked, thresholded binary:

*Mask*
![alt text][image6]

*Masked binary image*
![alt text][image7]

#### 3. Perspective transform

For the perspective transform I manually chose then hardcoded the following source and destination points. The transform 

```python
src_points = np.float32([
    [596, 449], # top-left
    [685, 449], # top-right
    [1054, 678], # bottom-right
    [254, 678] # bottom-left
])
destination_points = np.float32([
    [350, 0], # top-left
    [950, 0], # top-right
    [950, 678], # bottom-right
    [350, 678] # bottom-left
])
```
The transformation and inverse transformation matrices were found using:
```
M = cv2.getPerspectiveTransform(src_points, destination_points)
M_inv = cv2.getPerspectiveTransform(destination_points, src_points)
```
From which point images are transformed using `cv2.warpPerspective` for example:
```
img_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

*Source points*
![alt text][image12]

*Destination points*
![alt text][image13]

*Warped then unwarped*
![alt text][image14]

#### 4. Lane line pixel finding and fitting
Having got a warped binary image, the lane lines are found in 2 steps (see function `get_line_fits`):
    1. Base of the line is found using a histogram 
    2. Line is found using moving window method

For example returning to our warped binary image example from before:

*Masked binary image*
![alt text][image7]

*Histogram of bottom 1/3*
![alt text][image15]

*Lane lines*
![alt text][image8]

The lines were sanity checked by checking that the difference between the fit coefficients of the left and right lines did not exceded some maximum values. These maximum values were determined manually (with the starting point being the max diffs found in the subclips of the video where the lines were judged to be correct - in fact the full video pipeline was built before choosing these values, with lanes deemed to be "correct" outputed in green and "wrong" ones in red which enabled a quick feedback loop when tweaking these parameters).

#### 5. Radius of curvature and offset
The radius of curvature is calculated at the base of the lane lines (e.g. the bottom of the image):

```
def calculate_curvature(fit, y0):
    """
    For f(y) = Ay^2 + By + c
    R = (1 + (2Ay+B)^2)^(3/2) / abs(2A)
    """
    return (1 + (2*fit[0]*y0 + fit[1])**2)**(3/2) / np.abs(2*fit[0])
```

The offset is calculated by determining the position of the base of the left and right lane lines, averaging and subtracting the centre of the image (assuming the camera is mounted centrally on the vehicle).
```
def get_offset(img, fits):
    y0, w = img.shape
    lane_line_bases = [fit[0]*y0**2 + fit[1]*y0 + fit[2] for fit in fits]
    return (np.mean(lane_line_bases) - w/2) * xm_per_pix
 ```

#### 6. Final image
Here is the final image, which consists of the original image undistorted and overlayed with the filled in lane and information about the curvature, offset and line fits.

![alt text][image6]

---

### Pipeline (video)

#### 1. Video result

XXX Quick discussion of changes made for video

I tested using the margin from the previous fit line to find the lane line pixels but found that the original method of histogram + sliding windows gave better results (at least for my setup). However intuitively going off the previous fit feels like it ought to be a more robust method so perhaps a future improvement would be to dive deeper into this.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

Difficulties - mask for binary image, tweaking thresholds etc...

Challenge video
Harder challenge video

Weaknesses/where it might fail
Possible improvements/next steps
    Smoothing
    Convolution
    Better colour thresholds (perhaps hue specific for white/yellow)
    More time on gradient thresholds
    Sanity check could use change of fit from frame to frame (as well as between left/right lines)
