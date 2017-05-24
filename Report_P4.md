## Report Project 4 Advanced Lane Finding

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

[image1]: ./write_up_img/calibration_corners.jpg "calibration corners"
[image2]: ./write_up_img/undistorted_image.jpg "Undistorted image"
[image3]: ./write_up_img/image_pipeline/original.jpg "Example of image"
[image4]: ./write_up_img/image_pipeline/2_1_after_calibration.jpg "Undistorted image"
[image5]: ./write_up_img/image_pipeline/2_2_edge_detection.png "Edge detection"
[image6]: ./write_up_img/image_pipeline/2_2_edge_detection_illutration.jpg "Edge detection steps"
[image7]: ./write_up_img/image_pipeline/2_3_obtain_perspective_transform.png "Obtain perspective transform"
[image8]: ./write_up_img/image_pipeline/2_3_perspective_transform.png "After perspective transform"
[image9]: ./write_up_img/image_pipeline/2_3_binary_perspective_transform.png "Binary image after perspective transform"
[image10]: ./write_up_img/image_pipeline/2_4_lane.png "Center lane pixels before outleir rejection"
[image11]: ./write_up_img/image_pipeline/2_4_lane_reject_outlier.png "Center lane pixels after outleir rejection"
[image12]: ./write_up_img/image_pipeline/2_5_lane_single_fit.png "Seperate lane fit"
[image13]: ./write_up_img/image_pipeline/2_5_lane_joint_fit.png "Joint lane fit"
[image14]: ./write_up_img/image_pipeline/2_6_marked_lane_boundary.jpg "Marked lane boundary"
[image15]: ./write_up_img/marked_images/test1.jpg "Pipeline applied to test1.jpg"
[video1]: ./output_videos/project_video.mp4 "Video"


### Organization of Report
The code to produce results in this report is in "Advanced_Lane_Detection.ipynb"
We also reuse code from Project 1 when calculating the perspective transformation matrix. The functions are in "project1_module.py"

---


### 1. Camera Calibration

#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in code cells 1-7 in "Advanced_Lane_Detection.ipynb"

Here I assume that the same camera is used to capture all the provided chessboard images or at least the camera distortion is the same. To improve the calibration accuracy, we try to use as many as possible provided images. 

- We first search for registration points for calibration, which are the corners of the chessboard in the provided images. We mainly the `cv2.findChessboardCorners` for this task with some tweaks explained later this section.

- We then map each of the corners to a point in a rectangular grid. We collect all the corners and the mapped points.
- We then calculate the calibration parameter using `cv2.calibrateCamera` function.
- We then undistort the first image using the calibrated parameters via `cv2.undistort()`

##### Identified corners
The identified corners for the images are depicted below where the image title is the size of the grid identified for this image.

![alt text][image1]

Note that the number of corners in each image is not the same among the provided images, therefore, we try to repeatedly apply cv2.findChessboardCorners with different grid sizes, according to the following order [(9,6), (9,5), (8,6), (8,5), (7,6), (7,5), (6,6), (5,6)], and will return the corners the first time that cv2.findChessboardCorners is able to find them.


##### Undistorted image
The test image and the undistorted version is shown below:

![alt text][image2]


### 2. Pipeline (single image)

We use the following image as the example for the pipeline.

![alt text][image3]

#### 1 Provide an example of a distortion-corrected image.

We apply the calibratoin using the distortion coefficient obtained from section 1 to the example image. This is in code cell 10 where the fuction `pipline_undistort` is from code cell 6 and 7

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

We calculate gradient magnitude and direction based on the gray and S channel of the image, and use thresholding to obtain the binary image for the edges. The function is `edge_detection` in code cell 13 which uses `dir_abs_gradient_thresh` (cell 12) and `mag_gradient_thresh` (cell 11). 
The image we obtained is

![alt text][image5]

To be more precise, we apply the following steps:
- Convert image to gray scale to obtain A
- Use thresholding on gradient magnitudes of A to obtain binary image A1
- Use thresholding on gradient direction of A to obtain binary image A2
- Extract S channel of the image in its HSV representation to obtain B
- Use thresholding on gradient magnitudes of B to obtain binary image B1
- Use thresholding on gradient direction of B to obtain binary image B2
- The final image is obtained from applying the following pixel-wise logical operation: (A1 and A2) or (B1 and B2)
The steps are illustrated in the following image

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To apply perspective transform, we need to complete the following two steps:

- Obtain the matrix for perspective transform from the 'straing lines 1' image. (code cell 16-18)
- Apply the same perspective transform to new images. (code cell 22)

##### 3.1 Obtain perspective transform 

To calculate the perspective transform matrix, we apply `cv2.getPerspectiveTransform` which requires us to identify source and destination point sets, i.e., 4 points the original image and their location in the warped image. 

To obtain the 4 poitns in the original image, we applying the following steps:
- use the basic lane detection algorithm in Project 1 to identify two adjacent lanes in the image
- manually select two horizontal lanes that intersects these two lanes, which are the source points for calculation.
- Use the x location of the bottom two points ($x_1$, $x_2$) as the x location for the destination points and define the y location for the destination points manually.
- Calculate the transformation matrix `M_perspective_transform` using `cv2.getPerspectiveTransform`

Note that in order to not miss pixels when the lane is curving, we shrink the horizontal distance of the destination points: instead of using $x_1$, $x_2$ as the destination points's x location, we use $x_1+\delta$ and $x_2-\delta$ where $\delta=25$

The identified the source and destination points are depicted in 

![alt text][image7]

The calculated source and destination points are shown in code cell 19

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 525, 500      | 259, 450      | 
| 764, 500      | 1036, 450     |
| 234, 700      | 259, 700      |
| 1061, 700     | 1036, 700     |


##### 3.2 Apply perspective transform 
We apply the perspective transform using the matrix obtained from the above procedure to the binary images. Note that in writing this report, I use apply the transform to the same image 'straing lines 1'. In section 3, I applied the **same** perspective transform to other images.

The obtained binary image after the perspective transform are below

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is done in 3 steps:
- identify center lane pixels from the warped binary image which contains warped edge pixels. This is done by convoluting the image with a 2D one-valued arrays of size 21 x 61, and then find the pixel with the highest value per y location in the left and right half of the image. The function is `lane_identification` in code cell 25.
- reject outliers based on the x location of identified pixels. This is done by comparing pixel's x location to the **median** and rejects points that are outside of 5 times the **median absolute deviation**. The source of these outliers are either noise in edge detection or pixels from non-adjacent lanes.
- fit a polynomial to the identified center lane pixels after rejecting outliers. We experimented with two methods and decided to use a joint fitting method that we will describe below.

##### center pixel and outlier rejection
The identfied center pixels are depicted below:

![alt text][image10]

The center pixels after outlier rejection are depicted below

![alt text][image11]

##### lane fitting

It is very frequent to have limited number of marked lane segmented in the view. 
Initially we apply fit the quadratic polynomial to the center lane pixels on the left and center lane pixels on the right. However, this could become problematic when there are only 2 marked short lane segment for the right lane or for the left lane. The basic intuition is that when the lane segments are short, then each lane segments effectly become "one" point in the 2D image, and fitting a quaratic function with 3 degrees of freedom to 2 points is an ill-posed problem.  Although there are actually multiple points for each segment, when the segment is short, the variation in x and y values is not enough to provide a stable estimate. The fitting is still a ill-posed problem. 

To overcome this issue, we leverage the fact that the horizontal distance between the left lane and right lane is approximately a constant. In other words, the two quadratic functoin are identical except for a constant shift. Therefore, we fit one qudratic function to pixels from **both** lanes, adding one additional term to account for the shift. Mathematically speaking, we calculate the regression for the following formula:
$$ x = Ay^2 + By + C + g(x,y)$$ where $g(x,y) = -1$ for pixels $(x,y)$ from the left lane and $g(x,y) = 1$ for pixels from the right lane. 

The separate fit function is `get_curvature`, and the joint fit function is `get_curvature_joint_two_lanes` in code cell 30.

The comparision of the fitting results are below:
Result from separate fit:

![alt text][image12]

Result from join fit:

![alt text][image13]

We observe that the joint fit produces significantly more stable estimate. Note that we use y_center to shift the y position in order to reduce the multicollinearity in the regression.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and vehicle position calculation is also done in function `get_curvature_joint_two_lanes` defined in code cell 30.

I convert the x, y value of pixels to real world distances using ym_per_pix and xm_per_pix in code cell 32. We use the coefficient from the regression to calculate the radius of curvature in the following code
`Rcurve_car = (1+(2*(y_car_pos-y_center)*ym_per_pix*coefs[0]+coefs[1])**2)**(3.0/2)/np.abs(2*coefs[0])`

The position calculated by comparing the center of the image to the lane's x location at y = 700

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


I implemented this step in `mark_unwarped_lane` in code cell 35. 

![alt text][image14]

### Pipeline (multiple image)

I define the pipeline `pipeline` defined in code cell 37 that captures the above steps and applied to all test images. For the abbrevity of the report, I only show the pipeline result on a second image. The rest of the illustration can be found in 
I applied the pipline in `./write_up_img/marked_images/`
![alt text][image15]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)
<video controls src="./output_videos/project_video.mp4" />
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue with the implementation is the curvature calculation: Let's examine the equation for the curvature:
$$R_{curve} = \frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}$$
We note that this calculate is very sensitive to errrors in $A$ due to its role in the demoninator. On the other hand, the regression problem suffers from multicollinearity which makese it difficult to obtain a reliable estimate of $A$. To solve this issue, I would consider 
- reparameterization of the regression equation using the $R_{curve}$ parameter.
- regulerization in parameter estimate
- combine data from multiple frames

The second issue is the selection of perspective transform in which we decide how much of the lanes are covered in the unwarped image. By changing the specified y value of the corner points, we could include more or less farther-away portions of the lanes (pixels near the top of the image) But notice that the image noise and error in perspective transform is magnified for those pixels. This might be partly mitigated by performing a weighted regression instead of Ordinary Least Squares regression. 

The third issues that we detect lane edge pixel instead of the lane pixels in this pipeline. While we apply essentialy a low-pass filter to try to bridge the pixels, it is an ideal solution. One posssible improvment is to use the edge pixels as a references to extract pixels from gray scale and S channel of the image, for example, by only including high-value pixels surrounded by the edges. 