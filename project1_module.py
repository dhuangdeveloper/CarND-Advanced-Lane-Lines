import math
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import cv2

def draw_lines(img, lines, color=[255, 255, 0], thickness=2):
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
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def get_hough_parameter(x1,y1,x2,y2):
    """Calculate hough parameter
    
    Args:
        x1,y1,x2,y2: coordinates of the end points of the segment
    Returns:
        (rho, theta): Hough parameter
    """
    theta0 = math.atan2(x1-x2, y2-y1)/math.pi*180
    if theta0 >= 90:
        theta = theta0-180
    elif theta0 < -90:
        theta = theta0+180
    else:
        theta = theta0
    rho = x1 * math.cos(theta/180*math.pi) + y1 * math.sin(theta/180*math.pi)
    return rho, theta

def distance_in_hough_parameter(line1, line2, rho_weight = 10):
    """Two calculate pairwise distance, we use middle point as the origin to calculate hough parameter
    """
    
    
    x0 = np.median([line1[0], line1[2], line2[0], line2[2]])
    y0 = np.median([line1[1], line1[3], line2[1], line2[3]])
    x1l1 = line1[0]-x0
    y1l1 = line1[1]-y0
    x2l1 = line1[2]-x0
    y2l1 = line1[3]-y0
    x1l2 = line2[0]-x0
    y1l2 = line2[1]-y0
    x2l2 = line2[2]-x0
    y2l2 = line2[3]-y0
    rho1, theta1 = get_hough_parameter(x1l1, y1l1, x2l1, y2l1)
    rho2, theta2 = get_hough_parameter(x1l2, y1l2, x2l2, y2l2)
    return max(abs((rho2-rho1)/rho_weight), abs(theta1-theta2))
    

def cluster_and_combine_in_hough(lines, distance_criterion = 20, rho_weight_cluster=10, use_heuristic_distance=False):
    """Cluster the lines based on Hough transform
    
    Args: 
        lines (list of list of end points):
        distance_criterion (float) : maximum distance in Hough space for lines in the same cluster
    Returns:
        combined line parameters        
    """
    hough_parameter = np.stack([np.array(get_hough_parameter(x1,y1,x2,y2)) for line in lines for x1,y1,x2,y2 in line], axis=0)
    xy_matrix = np.stack([np.array([x1,y1,x2,y2]) for line in lines for x1,y1,x2,y2 in line], axis=0)        
    segment_length = np.array([math.hypot(x1-x2,y1-y2) for line in lines for x1,y1,x2,y2 in line])    
    if use_heuristic_distance: 
        dist_vector = pdist(xy_matrix, lambda u,v:distance_in_hough_parameter(u,v, rho_weight_cluster))
    else:
        dist_vector = pdist(hough_parameter/np.expand_dims(np.array([rho_weight_cluster,1]), axis=0), 'chebyshev')
    Z = linkage(dist_vector, method='average')
    cluster_assignment = fcluster(linkage(hough_parameter), t = distance_criterion, criterion='distance')
    # calculate lines for line in each cluster
    unique_cluster_number=np.unique(cluster_assignment)
    ave_lines = [[]]*len(unique_cluster_number)
    lines_in_hough = [[]]*len(unique_cluster_number)
    lines_in_hough_with_length = [[]]*len(unique_cluster_number)
    for jj in range(0, len(unique_cluster_number)):
        cluster_index = unique_cluster_number[jj]
        
        ave_rho_theta = np.average(hough_parameter[cluster_assignment==cluster_index,:],
                                   weights = (segment_length[cluster_assignment==cluster_index]-1)**1,
                                   axis=0)
        # weighted rho and theta by the length of each line
        endpoint_cord = np.concatenate([np.array([[x1,y1],[x2,y2]])for line in lines[cluster_assignment==cluster_index] for x1,y1,x2,y2 in line],
                           axis=0)
        ave_theta = ave_rho_theta[1]/180*math.pi
        ave_rho = ave_rho_theta[0]
        projected_position = endpoint_cord[:,0] * math.sin(ave_theta) - endpoint_cord[:,1] * math.cos(ave_theta)
        p1_projected = np.amax(projected_position)
        p2_projected = np.amin(projected_position)
        p1 = (p1_projected*math.sin(ave_theta)+ave_rho*math.cos(ave_theta), -p1_projected*math.cos(ave_theta)+ave_rho*math.sin(ave_theta))
        p2 = (p2_projected*math.sin(ave_theta)+ave_rho*math.cos(ave_theta), -p2_projected*math.cos(ave_theta)+ave_rho*math.sin(ave_theta))
        ave_lines[jj] = [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]]
        lines_in_hough_with_length[jj] = (ave_rho, ave_theta, math.hypot(p1[0]-p2[0],p1[1]-p2[1]))
        lines_in_hough[jj] = (ave_rho, ave_theta)
    return ave_lines, lines_in_hough, lines_in_hough_with_length

    # select the two lines with maximum lines
    #line_length = np.array([math.hypot(x1-x2,y1-y2) for line in ave_lines for x1,y1,x2,y2 in line])
    #max_line_indices = np.argsort(line_length)
    

#    return [ave_lines[index] for index in max_line_indices[-3:]], [lines_in_hough[index] for index in max_line_indices[-4:]]  

def select_lines(img, lines_in_hough_with_length, theta_limit, minimum_length, mid_view_interval):
    """ Select the middle two lines in view
    
    """
    # exclue out near horizontal lines and short lines
    #line_par = 
    #selected_lines = []
    selected_lines = [line_par for line_par in lines_in_hough_with_length if 
                      (line_par[1] >= theta_limit[0]) and (line_par[1] <= theta_limit[1]) and (line_par[2]>=minimum_length)]
    
    
    # calculate the intercepting point of lines with y_base_line = img.shape[1]
    y_base_line = img.shape[1]
    lines_with_intercepting_point = [(line_par,
                                      (line_par[0] - y_base_line * math.sin(line_par[1]))/math.cos(line_par[1])) for line_par in selected_lines]
    # divide lines into 3 groups (left, middle (exclude), right)
    left_lines = [line for line in lines_with_intercepting_point if line[1] <= mid_view_interval[0]]
    right_lines = [line for line in lines_with_intercepting_point if line[1] >= mid_view_interval[1]]
    # find the lines closest to the middle in the two groups (left and right)
    final_selected_lines = []
    if left_lines:
        left_line_selected = left_lines[np.argmax([line[1] for line in left_lines])][0]
        final_selected_lines = final_selected_lines + [(left_line_selected[0], left_line_selected[1])]
    if right_lines:
        right_line_selected = right_lines[np.argmin([line[1] for line in right_lines])][0]
        final_selected_lines = final_selected_lines + [(right_line_selected[0], right_line_selected[1])]        
    return final_selected_lines
    
def draw_lines_to_boundary_of_image(img, lines_in_hough, color=[255, 0, 0], thickness=2):
    """ Draw lines on a black image that go from boundary to boundary
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for (rho,theta,) in lines_in_hough:                
        if abs(theta)<=(math.pi/2):
            cv2.line(line_img,
                     (int(round(rho / math.cos(theta))), 0),
                     (int(round((rho - line_img.shape[1]*math.sin(theta)) / math.cos(theta))), line_img.shape[1]),
                     color, thickness)
        else:
            cv2.line(line_img,
                     (0, int(round(rho / math.sin(theta)))),
                     (int(round(line_img.shape[0], (rho - line_img.shape[0]*math.cos(theta)) / math.sin(theta))), line_img.shape[1]),
                     color, thickness)
    return line_img
                               

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


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def process_image(image, 
                  distance_criterion=10, 
                  rho_weight_cluster=3, 
                  use_heuristic_distance=False, 
                  y_view_percentage=0.63):
    """
        Fine the two closest lines in the image
    """


    # blured image
    #gray_smoothed = gaussian_blur(image_gray_scale, 5)

    
    # get region of interest
    vertices = np.array([[(0, image.shape[0]), (int(image.shape[1] *0.4),  int(image.shape[0]*y_view_percentage)),(int(image.shape[1] *0.6), int(image.shape[0]*y_view_percentage)), (image.shape[1],image.shape[0])]], dtype=np.int32)
    mask = region_of_interest(image, vertices)

    # mask edges
    masked_edges = cv2.bitwise_and(image, mask)
    

    # detect lines
    lines_detected = cv2.HoughLinesP(masked_edges, 
                            2, 
                            np.pi/180, 
                            60, 
                            np.array([]), 
                            minLineLength=200, 
                            maxLineGap=300)
    #immage2= cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #draw_lines(immage2, lines_detected, color=[0, 255, 0], thickness=2)# to be removed

    # cluster lines
    ave_lines, ave_line_in_hough, lines_in_hough_with_length = cluster_and_combine_in_hough(lines_detected,
                                                                                    distance_criterion=distance_criterion,
                                                                                    rho_weight_cluster=rho_weight_cluster,
                                                                                    use_heuristic_distance=use_heuristic_distance)
    # down select lines
    down_selected_lines = select_lines(image, lines_in_hough_with_length, (-60.0/180*math.pi, 60.0/180*math.pi), 
                                       minimum_length = 100, mid_view_interval=(image.shape[1]/2*0.99, image.shape[1]/2*1.01))

    return down_selected_lines#, immage2