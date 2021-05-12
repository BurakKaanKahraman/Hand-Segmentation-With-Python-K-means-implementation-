# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:48:21 2021

@author: kaankhrmn
"""
import os
import numpy as np
from PIL import Image
import glob
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2
import math
from scipy import signal
import random
from collections import Counter
from statistics import mode

Inputs = r"C:\Users\kaankhrmn\Desktop\Comp Vision\Dataset"
Outputs = r"C:\Users\kaankhrmn\Desktop\Comp Vision\Dataset\SegmentationResults\*"
Outputs2= r"C:\Users\kaankhrmn\Desktop\Comp Vision\Dataset\NormResults"

def NormRGB (Input):
    Output2 = Input+"\Results"
    Inputt = Input+"\Dataset"
    # Taking all the images in the Dataset directory
    images = []
    result_images=[]

    for filename in glob.glob(Inputt + '\\*.png'):
        im=Image.open(filename)
        images.append(im)

    count=1
    # After takin all the images we will traverse each one of them and proccess them
    for image in images:
        #image = Image.open('1.png')  # Taking image from the dir
        image=  image.convert('RGB')
        #checking the format size and the model of the image
        #print(image.format)
        #print(image.size)
        #print(image.mode)

        #converting image into numpy array 3d matrix
        image_array = np.asarray(image)

        #print(image_array.shape)

        h = image_array.shape[1]  # We get number of rows (y)
        w = image_array.shape[0]  # We get number of colums (x)

        new_img = np.zeros((h, w, 3),
                           dtype=np.int8)  # We are creating New İmgae as matrix consists of 0 for width and height for each RGB value

        print(new_img.shape)

        # We are getting all the pixel values by visiting each pixel by height widht and the layer(r=0,g=1,b=2)
        for x in range(0, h):  # for each height
            for y in range(0, w):  # for each widht

                # We are taking each pixels red green blue values to proccess
                Colors =image.getpixel((x,y))
                red = Colors[0]
                green = Colors[1]
                blue = Colors[2]

                # For calculating ratio of the specific color to sum of them (red/total vs.....)
                total_value = float(red + green + blue)


                #r = (red / total_value) * red
                r = (red / total_value) *red
                g = (green / total_value) *green
                b = (blue / total_value) *blue

                #r = (red / total_value) * 255
                #g = (green / total_value) * 255
                #b = (blue / total_value) * 255

                # Now we are assigning the new values to the created zero matrix to form new image
                new_img[y, x, 0] = r
                new_img[y, x, 1] = g
                new_img[y, x, 2] = b

        # We are converting numpy array to PIL Image object to reach new image

        show_img = Image.fromarray((new_img*255).astype(np.uint8))
        # We save new image to the directory
        try:
            os.makedirs(Input+"\Results")
            print("Directory ", Output2, " Created ")
        except FileExistsError:
            pass
        show_img.save(Output2+'/'+'new_img'+str(count)+'.png')
        count=count+1
        # We are showing new image
        #show_img.show()



# This function calculates the euclidian distance
def Euclidian_distance(x1,x2,y1,y2,z1,z2):
    dst = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
    #dst = distance.euclidean(Point_x, Point_y)
    return dst



def Kmeans(iteration,image,output_name=""):

    # Getting number of rows and colums of the images
    max_x = image.shape[0]
    max_y = image.shape[1]
    # print(max_x,max_y)

    # We are getting seeds and How many cluster(K) there should be from the claculate_k() function
    # red green and blue represents the peak values of the image for each color
    # peak represents how many cluster(K) should be
    red, green, blue, peak = calculate_k(image)

    Red_peak_info = red
    Green_peak_info = green
    Blue_peak_info = blue
    K= peak

    # We are shuffling peak values for getting better result(we prevent matching same values (low red- low blue all the time))
    random.shuffle(Red_peak_info)
    #print("Red_peak_info : {}".format(Red_peak_info))
    random.shuffle(Green_peak_info)
    random.shuffle(Blue_peak_info)
    # Creating new matrix for the new image
    image_array = np.asarray(image)

    # We hold every centroids pixel values in centroid_dimension[R,G,B]
    centroid_dimension=[0,0,0]

    # Centroid_point_lists holds points of the each cluster
    Centroid_point_lists=[[] for _ in range(K) ]

    #Centroid_list Holds the centroids
    Centroid_list=[]
    Centroid_dimension_list=[[0,0,0] for _ in range(K) ]
    #print(Centroid_dimension_list)



    # For Each cluster K
    for i in range (K):

        #We are creating cluster with given seeds (Seeds are taken from calculate_k() function. This function returns peak pixel values for each color and K)
        Centroid = [Red_peak_info[i],Green_peak_info[i],Blue_peak_info[i]]
        Centroid_list.append(Centroid)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(Centroid_list)
    # Outher for loop represents the iteration number (How many times we will repat the proccess)
    for turn in range(iteration):

        # we will get each pixels R G B values to calculate distance and update our centroid
        for x in range(max_x):
            for y in range(max_y):
                # We are getting RGB values of the spesific point
                point = image[x, y]
                point_final = [x, y]
                dist_total = []
                #print("====================================>")
                #print("Point {}".format(point_final))

                # Calculating distance of the point to each centroid points
                for centroid in Centroid_list:
                    cp1 = (centroid[0])
                    cp2 = (centroid[1])
                    cp3 = (centroid[2])
                    #cp3 = np.all(Centroid_list[i][2])
                    #print("Distance point 1 : {} Distance point 2 : {} Distance point 3 : {} ".format(Centroid_list[i][0],Centroid_list[i][1],Centroid_list[i][2]))
                    #dist = Euclidian_distance(point[0], Centroid_list[i][0] , point[1], Centroid_list[i][1], point[2], Centroid_list[i][2] )
                    dist = Euclidian_distance(point[0], cp1, point[1], cp2, point[2],cp3)
                    dist_total.append(dist)

                # We are appending each distance to distance list. Then we will find the index of the min distance to find which centroid is closest
                min_dist = min(dist_total)
                dist_index = dist_total.index(min_dist)

                #print("Len of the dist_total =======> {}".format(len(dist_total)))
                #Centroid_point_lists[dist_index].append(point_final)
                #print("Index  =======> {}".format(point_final))


                # After finding closest centroid we append this point to centroids own point list
                Centroid_point_lists[dist_index].append(point_final)
                #print("Added Centroid_Point ===========> {}".format(Centroid_point_lists[dist_index]))
                #print("Cetnroid_List : {} {} {}".format(Centroid_dimension_list[dist_index],Centroid_dimension_list[dist_index],Centroid_dimension_list[dist_index]))


                # We are adding each pixel value to update our centroids later
                Centroid_dimension_list[dist_index][0] = Centroid_dimension_list[dist_index][0] + point[0]
                Centroid_dimension_list[dist_index][1] = Centroid_dimension_list[dist_index][1] + point[1]
                Centroid_dimension_list[dist_index][2] = Centroid_dimension_list[dist_index][2] + point[2]

        # We are updating each centroid for the next iteration
        for cluster in range(K):
            #print("Centoid_point List : {} ".format(len(Centroid_point_lists[0])))
            denominator = len(Centroid_point_lists[cluster])
            if(denominator <= 0):
                denominator=1
            if(len(Centroid_list[cluster])>0):
                Centroid_list[cluster] = [Centroid_dimension_list[cluster][0]/denominator,Centroid_dimension_list[cluster][1]/denominator,Centroid_dimension_list[cluster][2]/denominator]
            print("===================================")
            print("New Cluster{} is : {}".format(cluster, Centroid_point_lists[cluster]))

        # Until the last iteration we are emptying our Centroid Point List to prevent duplicate points
        if(turn!=iteration-1):
            Centroid_dimension_list=[[0,0,0] for _ in range(K) ]
            Centroid_point_lists=[[]for _ in range(K)]

        print("Iteration {} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".format(turn+1))

    #Show and Save image
    """
    # We are converting numpy array to PIL Image object to reach new image
    show_img = Image.fromarray((image_array), 'RGB')
    try:
        os.makedirs(Output)
        print("Directory ", Output, " Created ")
    except FileExistsError:
        pass
    # We save new image to the directory
    show_img.save(Output + '/' + output_name + '.png')
    # We are showing new image
    #show_img.show()
    """
    return Centroid_point_lists

# Its the same function But this will show the histograms to find peaks (Its for debug purposes)
def Kmeans_with_histogram_show(iteration,image,output_name=""):

    # Getting number of rows and colums of the images
    max_x = image.shape[0]
    max_y = image.shape[1]
    # print(max_x,max_y)

    # We are getting seeds and How many cluster(K) should be from the claculate_k() function
    # red green and blue represents the peak values of the image for each color
    # peak represents how many cluster(K) should be
    red, green, blue, peak = calculate_k_with_histogram(image)

    Red_peak_info = red
    Green_peak_info = green
    Blue_peak_info = blue
    K= peak

    # Creating new matrix for the new image
    image_array = np.asarray(image)
    centroid_dimension=[0,0,0]
    Centroid_point_lists=[[] for _ in range(K) ]

    Centroid_list=[]
    Centroid_dimension_list=[[0,0,0] for _ in range(K) ]
    #print(Centroid_dimension_list)




    for i in range (K):
        Centroid = [Red_peak_info[i],Green_peak_info[i],Blue_peak_info[i]]
        Centroid_list.append(Centroid)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(Centroid_list)
    # Outher for loop represents the iteration number (How many times we will repat the proccess)
    for turn in range(iteration):

        # we will get each pixels R G B values to calculate distance and update our centroid

        for x in range(max_x):
            for y in range(max_y):
                # We are getting RGB values of the spesific point
                point = image[x, y]
                point_final = [x, y]
                dist_total = []
                #print("====================================>")
                #print("Point {}".format(point_final))

                # Calculating distance of the point to each centroid points
                for centroid in Centroid_list:
                    cp1 = (centroid[0])
                    cp2 = (centroid[1])
                    cp3 = (centroid[2])
                    #cp3 = np.all(Centroid_list[i][2])
                    #print("Distance point 1 : {} Distance point 2 : {} Distance point 3 : {} ".format(Centroid_list[i][0],Centroid_list[i][1],Centroid_list[i][2]))
                    #dist = Euclidian_distance(point[0], Centroid_list[i][0] , point[1], Centroid_list[i][1], point[2], Centroid_list[i][2] )
                    dist = Euclidian_distance(point[0], cp1, point[1], cp2, point[2],cp3)
                    dist_total.append(dist)

                min_dist = min(dist_total)
                dist_index = dist_total.index(min_dist)
                #print("Len of the dist_total =======> {}".format(len(dist_total)))

                #Centroid_point_lists[dist_index].append(point_final)
                #print("Index  =======> {}".format(point_final))
                Centroid_point_lists[dist_index].append(point_final)
                #print("Added Centroid_Point ===========> {}".format(Centroid_point_lists[dist_index]))
                #print("Cetnroid_List : {} {} {}".format(Centroid_dimension_list[dist_index],Centroid_dimension_list[dist_index],Centroid_dimension_list[dist_index]))

                Centroid_dimension_list[dist_index][0] = Centroid_dimension_list[dist_index][0] + point[0]
                Centroid_dimension_list[dist_index][1] = Centroid_dimension_list[dist_index][1] + point[1]
                Centroid_dimension_list[dist_index][2] = Centroid_dimension_list[dist_index][2] + point[2]

                #print((Centroid_list[i][0]))

        for cluster in range(K):
            #print("Centoid_point List : {} ".format(len(Centroid_point_lists[0])))
            Centroid_list[cluster] = [Centroid_dimension_list[cluster][0]/(len(Centroid_point_lists[cluster])),Centroid_dimension_list[cluster][1]/(len(Centroid_point_lists[cluster])),Centroid_dimension_list[cluster][2]/(len(Centroid_point_lists[cluster]))]
            print("===================================")
            print("New Cluster{} is : {}".format(cluster, Centroid_point_lists[cluster]))

        if(turn!=iteration-1):
            Centroid_dimension_list=[[0,0,0] for _ in range(K) ]
            Centroid_point_lists=[[]for _ in range(K)]





        # Updating centroid points for next iteration
        #centroid1 = [(centroid_dimensions1[0]/len(centroid_point_list1)),(centroid_dimensions1[1]/len(centroid_point_list1)),(centroid_dimensions1[2]/len(centroid_point_list1))]
        print("Iteration {} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".format(turn+1))

    for point in Centroid_point_lists[0]:
        #print (image_array[point[0], point[1]])
        image_array[point[0], point[1]] = [255, 0, 0]
    for point in Centroid_point_lists[1]:
        #print (image_array[point[0], point[1]])
        image_array[point[0], point[1]] = [0, 0, 255]
    if(K>=3):
        for point in Centroid_point_lists[2]:
            # print (image_array[point[0], point[1]])
            image_array[point[0], point[1]] = [0, 255, 255]



    # We are converting numpy array to PIL Image object to reach new image
    show_img = Image.fromarray((image_array), 'RGB')
    # We save new image to the directory
    show_img.save(r"C:\Users\kaankhrmn\Desktop\Comp Vision\Dataset\SegmentationResults" + '/' + output_name + '.png')
    # We are showing new image
    show_img.show()

    #point = ["X","Y"]
    #Centroid_list[0].append(point)
    #print(Centroid_list)

    # Creating K times centroid


# With this function we are appliying Cluster algorith to spesific image as 3 Cluster (K=3). In this function K is fixed. We can change iteration variable to adjust iteration number.
def Kmeans_fixed(iteration,image):

    # In centroid Point List we are holding the list of points closer to centroid1
    centroid_point_list1 = []
    # In centroid dimensions we are holding the total value of the each color value (R,G,B) to adjust centroid later by dividing the centroid number each centroid has (Total Color Value/ Centroid Number vs...)
    centroid_dimensions1 = [0,0,0]

    centroid_point_list2 = []
    centroid_dimensions2 = [0, 0, 0]

    centroid_point_list3 = []
    centroid_dimensions3 = [0, 0, 0]

    # Getting number of rows and colums of the images
    max_x = image.shape[0]
    max_y = image.shape[1]
    # Creating new matrix for the new image
    image_array = np.asarray(image)

    # Obtaining and assigning random points to the centroids for initial centroid values
    rand_x = np.random.randint(0, max_x - 1)
    rand_y = np.random.randint(0, max_y - 1)
    centroid1 = [rand_x,rand_y]
    centroid1 = image[centroid1[0],centroid1[1]]

    rand_x = np.random.randint(0, max_x - 1)
    rand_y = np.random.randint(0, max_y - 1)
    centroid2 = [rand_x,rand_y]
    centroid2 = image[centroid2[0], centroid2[1]]

    rand_x = np.random.randint(0, max_x - 1)
    rand_y = np.random.randint(0, max_y - 1)
    centroid3 = [rand_x,rand_y]
    centroid3 = image[centroid3[0], centroid3[1]]

    # Outher for loop represents the iteration number (How many times we will repat the proccess)
    for turn in range(iteration):
        # we will get each pixels R G B values to calculate distance and update our centroid
        for x in range(max_x):
            for y in range(max_y):
                # We are getting RGB values of the spesific point
                point = image[x,y]
                point_final = [x,y]
                dist_total=[]

                # Calculating distance of the point to each centroid points
                d1 = Euclidian_distance(point[0], centroid1[0], point[1], centroid1[1], point[2], centroid1[2])
                d2 = Euclidian_distance(point[0], centroid2[0], point[1], centroid2[1], point[2], centroid2[2])
                d3 = Euclidian_distance(point[0], centroid3[0], point[1], centroid3[1], point[2], centroid3[2])

                dist_total=[d1,d2,d3]

                #print("============>")
                #print(dist_total)
                #print("============>")

                # Deciding which centroid closest to the point
                if min(dist_total)== d1:
                    # Adding point to Centroids point list
                    centroid_point_list1.append(point_final)
                    # Adding Each color value to previous value to get the total
                    centroid_dimensions1[0] = centroid_dimensions1[0]+point[0]
                    centroid_dimensions1[1] = centroid_dimensions1[1] + point[1]
                    centroid_dimensions1[2] = centroid_dimensions1[2] + point[2]
                    if (turn == iteration-1):
                        pass
                        #image_array[point_final[0],point_final[1]] = [255,0,0]
                    # image_array[point_final[0],point_final[1],0] = 255
                elif min(dist_total)== d2:
                    centroid_point_list2.append(point_final)
                    centroid_dimensions2[0] = centroid_dimensions2[0] + point[0]
                    centroid_dimensions2[1] = centroid_dimensions2[1] + point[1]
                    centroid_dimensions2[2] = centroid_dimensions2[2] + point[2]
                    if (turn == iteration-1):
                        pass
                        #image_array[point_final[0], point_final[1]] = [0, 255, 0]
                    # image_array[point_final[0],point_final[1],1] = 255
                elif min(dist_total)== d3:
                    centroid_point_list3.append(point_final)
                    centroid_dimensions3[0] = centroid_dimensions3[0] + point[0]
                    centroid_dimensions3[1] = centroid_dimensions3[1] + point[1]
                    centroid_dimensions3[2] = centroid_dimensions3[2] + point[2]
                    if (turn == -1):
                        pass
                        #image_array[point_final[0], point_final[1]] = [0, 0, 255]
                    # image_array[point_final[0],point_final[1],2] = 255

        #print("___________________________________________________")
        #print("Centroid 1 List is : {}".format(centroid_point_list1))

        #print("___________________________________________________")
        #print("Centroid 2 List is : {}".format(centroid_point_list2))

        #print("___________________________________________________")
        #print("Centroid 3 List is : {}".format(centroid_point_list3))

        # Updating centroid points for next iteration
        centroid1 = [(centroid_dimensions1[0]/len(centroid_point_list1)),(centroid_dimensions1[1]/len(centroid_point_list1)),(centroid_dimensions1[2]/len(centroid_point_list1))]

        centroid2 = [(centroid_dimensions2[0] / len(centroid_point_list2)),
                     (centroid_dimensions2[1] / len(centroid_point_list2)),
                     (centroid_dimensions2[2] / len(centroid_point_list2))]

        centroid3 = [(centroid_dimensions3[0] / len(centroid_point_list3)),
                     (centroid_dimensions3[1] / len(centroid_point_list3)),
                     (centroid_dimensions3[2] / len(centroid_point_list3))]

        print("================> ")
        print("Centoid1 : {}   Centroid2 : {}   Centroid3 : {}".format(centroid1,centroid2,centroid3))
        print("================> ")


    # We are converting numpy array to PIL Image object to reach new image
    show_img = Image.fromarray((image_array),'RGB')
    # We save new image to the directory
    show_img.save(r"C:\Users\kaankhrmn\Desktop\Comp Vision\Dataset\SegmentationResults" + '/' + 'new_cluster'+'.png')
    # We are showing new image
    show_img.show()

# This function calculates the peaks and the peak values of the given array/histogram
def calculate_peak(histogram):
    hist = np.array(histogram)  # convert your 1-D array to a numpy array if it's not, otherwise omit this line
    peak_gap = np.arange(1, 20) # Peak width found by trial and error (20 Best suits for finding distinct Peaks) (Basically we are checking every 20 pixel (1-20, 40-60 vs.....)) We can get max K=13
    peak_indices = signal.find_peaks_cwt(hist, peak_gap) # This function Will return the peak pixel values for each color
    #print(peak_indices)
    peak_count = len(peak_indices)  # We are counting the peak values to find peak number (the number of peaks in the array)
    peak_info=[peak_indices,peak_count]
    return peak_info

# In this function we are chechking our peak_count list and we are getting most common element in the peak count list to decide cluster number(K)
# If none of the elemets are same in the list then we pick the smallest one for K (Because We have to get the smallest to meet in common denominator)
# hist_peak_count is where we hold each peak value for each of the color (Red, Green , Blue)
def most_common_peak_count(hist_peak_count):
    # This function finds most common element in the list in this case most common peak count
    try:
        peak_count = max(set(hist_peak_count), key = hist_peak_count.count)
        # if the most common peak is bigger than the least numerious peak count than we should set the peak count to minimum
        if (peak_count > min(hist_peak_count) ):
            peak_count = min(hist_peak_count)
        #peak_count = mode(hist_peak_count)
    # If we cant find most common element or all of the counts are different we will get the smallest peak_count for our peak number
    except:
        peak_count = min(hist_peak_count)

    return peak_count

def calculate_k(image):
    # First We need to find histogram of the image
    # channels= 1=Red 2=Green 3= Blue
    h = image.shape[0]
    w = image.shape[1]
    # Empty Array for histogram
    Red_Hist = [0]*256
    Green_Hist = [0] *256
    Blue_Hist = [0] *256

    # We visit each pixel to get the R G B values to create histogram
    for x in range(0, h):
        for y in range(0, w):
            point = image[x, y]
            Red_Hist[point[0]] += 1
            Green_Hist[point[1]] += 1
            Blue_Hist[point[2]] += 1

    # We are calculating peak values and peak number for each color
    Red_peak_info = calculate_peak(Red_Hist)
    Green_peak_info = calculate_peak(Green_Hist)
    Blue_peak_info = calculate_peak(Blue_Hist)
    Peak_List=[]

    Peak_List.append(Red_peak_info[1])
    Peak_List.append(Green_peak_info[1])
    Peak_List.append(Blue_peak_info[1])


    print("===================>")
    print("Red Peak Number : => {}, Green Peak Number : => {}, Blue Peak Number : => {}".format(Red_peak_info[1],Green_peak_info[1],Blue_peak_info[1]))
    print("===================>")

    # We find optimal K count
    peak = most_common_peak_count(Peak_List)

    print("===================>")
    print("Optimal K is => {}".format(peak))
    print("===================>")

    # Will retun Peaked color for each centroid and max cluster number K
    calculated_histogram = []
    calculated_histogram= [Red_peak_info[0],Green_peak_info[0],Blue_peak_info[0],peak]

    Histogram=[]
    Histogram.append(Red_Hist)
    Histogram.append(Green_Hist)
    Histogram.append(Blue_Hist)

    return Red_peak_info[0],Green_peak_info[0],Blue_peak_info[0],peak

# Same with above but there is a histogram to debug
def calculate_k_with_histogram(image):
    # First We need to find histogram of the image
    # channels= 1=Red 2=Green 3= Blue
    h = image.shape[0]
    w = image.shape[1]
    Red_Hist = [0]*256
    Green_Hist = [0] *256
    Blue_Hist = [0] *256

    for x in range(0, h):
        for y in range(0, w):
            point = image[x, y]
            Red_Hist[point[0]] += 1
            Green_Hist[point[1]] += 1
            Blue_Hist[point[2]] += 1

    Red_peak_info = calculate_peak(Red_Hist)
    Green_peak_info = calculate_peak(Green_Hist)
    Blue_peak_info = calculate_peak(Blue_Hist)
    Peak_List=[]

    Peak_List.append(Red_peak_info[1])
    Peak_List.append(Green_peak_info[1])
    Peak_List.append(Blue_peak_info[1])


    print("===================>")
    print("Red Peak Number : => {}, Green Peak Number : => {}, Blue Peak Number : => {}".format(Red_peak_info[1],Green_peak_info[1],Blue_peak_info[1]))
    print("===================>")

    peak = most_common_peak_count(Peak_List)

    print("===================>")
    print("Optimal K is => {}".format(peak))
    print("===================>")

    # Will retun Peaked color for each centroid and max cluster number K
    calculated_histogram = []
    calculated_histogram= [Red_peak_info[0],Green_peak_info[0],Blue_peak_info[0],peak]

    Histogram=[]
    Histogram.append(Red_Hist)
    Histogram.append(Green_Hist)
    Histogram.append(Blue_Hist)

    plt.figure()
    plt.title("Red Histogram")
    plt.xlabel("Value")
    plt.ylabel("# of Pixels")
    plt.xlim([0,256])
    plt.plot(Blue_Hist)
    plt.show()


    return Red_peak_info[0],Green_peak_info[0],Blue_peak_info[0],peak


# After finding Points for each Centroid we need to diferantiate skin color to other colors
# We apply mask to differantiate skin to other objects by givin [0,0,0] to other clusters
def Masking (cluster_point_list,image,Output,output_name=""):

    # Defining variables to get total color values
    color_value=0
    total_color_value=[]
    image_array = np.asarray(image)

    # For each centroid
    for cluster in cluster_point_list:
        color_value = 0
        # We traverse its points to get the total color points
        for point in cluster:
            # We add all the pixel values Because in our normalization skin color is near to [200-255,200-255,180-210]
            # So if we add all the pixel values and find the cluster that has most the color point we can differantiate skin from other parts
            point_values = image_array[point[0], point[1]]
            color_value = float(point_values[0]) + float(point_values[1])+float(point_values[2])

        # We append color points to the list to find index
        total_color_value.append(color_value)
        print("=================>")
        print("Total color values : {}".format(total_color_value))
        print("=================>")
            #image_array[point[0], point[1]] = [0, 255, 255]

    # We are finding the centroid that has most of the color points
    max_value = max(total_color_value)
    index = total_color_value.index(max_value)

    # We are seperating this centroid from others
    del cluster_point_list[index]

    # Now we are painting rest of the points to the black [0,0,0] 
    for p in cluster_point_list:
        for point in p:
            image_array[point[0], point[1]] = [0, 0, 0]


    # We are saving the image to new directory
    show_img = Image.fromarray((image_array), 'RGB')
    # We save new image to the directory
    try:
        os.makedirs(Output)
        print("Directory ", Output, " Created ")
    except FileExistsError:
        pass
        # We save new image to the directory
    show_img.save(Output + '/' + output_name + '.png')
    # We are showing new image
    #show_img.show()

# This functions is for taking images from a directory
def get_all_images(folder, ext):
    all_files = []
    # Iterate through all files in folder
    for file in os.listdir(folder):
        # Get the file extension
        _, file_ext = os.path.splitext(file)

        # If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    # Get list of all files
    return all_files

if __name__ == "__main__":
    images = []

    # Getting Input and modifiying to further uses
    Input= input("Please enter the path")
    Input2 = Input+'\Results'
    Output=Input+ "\SegmentationResults"

    # Calling NormRGB()Function to normalise our images
    NormRGB(Input)

    #Getting all the normalised images to one list
    img_list=get_all_images(Input2,'png')
    #glob.glob(Input):
    for filename in img_list:
        im = cv2.imread(filename)
        images.append(im)

    #print("images : {}".format(images))


    #Usage: kmeans_fixed(iteration,image):
    # ==>kmeans_fixed(7,image)

    # For every image we are finding optimal K value and RGB values for seeds
    counter=0
    for image in images:

        kmeans_output = "Kmeans"+str(counter)
        masking_output = "Masking"+str(counter)
        counter = int(counter)+1
        centroid_points = Kmeans(5, image,kmeans_output)
        Masking(centroid_points, image, Output,masking_output)
    #Kmeans_with_histogram_show(3,images[2],"New image")
    #centroid_points=Kmeans(3,images[4],"Naber3")
    #Masking(centroid_points,images[4], "Sonunda3")

    # There are two or three images that is not converted properly its because of my function finds Optimal value of K=1 for these images so its not converting.
    # The reson for K=1 is some times Peak values are like that R=3 G=3 B=1 in this case we need to chose K=1 because We have to get the smallest to meet in common denominator.