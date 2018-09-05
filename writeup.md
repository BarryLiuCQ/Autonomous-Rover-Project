## Project: Search and Sample Return
---
**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid2.jpg
[image3]: ./calibration_images/example_rock2.jpg 
[image4]: ./output/perspective.png 
[image5]: ./output/warped_threshed.jpg
[image6]: ./output/rocks.png
[image7]: ./output/coords_transform.jpg
## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

![alt text][image2]
![alt text][image3]

#### 2. Perspective Transform
Using CV2 to perform Perspective transform,having a mask that leave only the Field of View (FOV) which is useful for applying to the obstacle image.

```python
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))#show field view of the camera
    return warped, mask
```
![alt text][image4]

#### 3. Color Thresholding
Apply color thresholding for both warped image(RGB>160) and rocks(R > 110 , G > 110 , B < 50)

for Warped image:
```python
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    color_select[above_thresh] = 1
    return color_select
threshed = color_thresh(warped)
```
for Rocks:
```python
def find_rocks(img, levels=(110, 110, 50)):
    color_select = np.zeros_like(img[:,:,0])
    rockpix = (img[:,:,0] > levels[0]) \
                & (img[:,:,1] > levels[1]) \
                & (img[:,:,2] < levels[2])
     color_select[rockpix] = 1
     return color_select
```
![alt text][image5]
![alt text][image6]
#### 4. Coordinate Transformations
(1)Extract x,y positions from color threshed image and convert to rover coords.(2)Convert rover coords to polar coords to calculate distance and angle.(3)Map rover space pixels to world space
```python
def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

def rotate_pix(xpix, ypix, yaw):
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    return xpix_translated, ypix_translated


def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    return x_pix_world, y_pix_world
```
![alt text][image7]


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
#### perception_step()

The perception step is basically the same as the notebook, with the only difference of upadating the Rover Worldmap by using"Rover.worldmap[y_world,x_world,2] += 10  Rover.worldmap[obs_y_world,obs_x_world,0] += 1".
```python
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    dst_size = 5
    bottom_offset = 6
    image = Rover.img
    source = np.float32([[14,140],[301,140],[200,96],[118,96]])
    destination = np.float32([[image.shape[1]/2 - dst_size,image.shape[0] - bottom_offset],
                              [image.shape[1]/2 + dst_size,image.shape[0] - bottom_offset],
                              [image.shape[1]/2 + dst_size,image.shape[0] - 2*dst_size - bottom_offset],
                              [image.shape[1]/2 - dst_size,image.shape[0] - 2*dst_size - bottom_offset],
                              ])
    warped,mask = perspect_transform(Rover.img,source,destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed= color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed)-1)*mask
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = threshed * 255
    Rover.vision_image[:,:,0] = obs_map * 255
    # 5) Convert map image pixel values to rover-centric coords
    xpix,ypix = rover_coords(threshed)
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2*dst_size
    x_world,y_world = pix_to_world(xpix,ypix,Rover.pos[0],Rover.pos[1],
                                   Rover.yaw,world_size,scale)
    obsxpix,obsypix = rover_coords(obs_map)
    obs_x_world,obs_y_world = pix_to_world(obsxpix,obsypix,Rover.pos[0],Rover.pos[1],
                                   Rover.yaw,world_size,scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[y_world,x_world,2] += 10
    Rover.worldmap[obs_y_world,obs_x_world,0] += 1
    # 8) Convert rover-centric pixel positions to polar coordinates
    dist,angles = to_polar_coords(xpix,ypix)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_angles = angles
    
    #find rocks
    rock_map = find_rocks(warped,levels=(110,110,50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world,rock_y_world = pix_to_world(rock_x,rock_y,Rover.pos[0],
                                                 Rover.pos[1],Rover.yaw,world_size,scale)
        rock_dist,rock_ang = to_polar_coords(rock_x,rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
 
        Rover.worldmap[rock_ycen,rock_xcen,1] = 255
        Rover.vision_image[:,:,1] = rock_map*255
    else:
        Rover.vision_image[:,:,1] = 0
    return Rover
```
#### decision_step()


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  


**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]
