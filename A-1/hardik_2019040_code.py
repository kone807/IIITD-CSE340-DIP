#!/usr/bin/env python
# coding: utf-8

# # DIP A-1 (coding questions)
# 
# ## Submitted By : Hardik Garg, 2019040

# ### Q3 - Bilinear Interpolation

# In[20]:


import numpy as np, cv2
from math import *

def bilinear_interpolation(image, c):
    
    M1, N1 = image.shape
    M2, N2 = ceil(M1*c), ceil(N1*c)
    
    # initialize output matrix, -1 serves as a dummy value
    final_image = np.full((M2,N2),-1)
    
    # map input grid to output grid
    
    for i in range(M1):
        for j in range(N1):
            final_image[int(i*c)][int(j*c)]=image[i][j]
            
    Mx = int(i*c)
    Ny = int(j*c)
    
    # iterating over values in output matrix corresponding to input matrix
    
    for i in range(Mx+1):
        for j in range(Ny+1):
            
            if final_image[i][j]==-1:
                
                # find 4 nearest neighbours corresponding to input matrix
                
                x,y = i/c, j/c
                
                # x1<x2 and y1<y2
                
                if ceil(x)!=x:
                    x1=floor(x)
                    x2=ceil(x)
                    
                else:
                    if x==0:
                        x1=0
                        x2=1
                    else:
                        x1=x-1
                        x2=x
                
            
                if ceil(y)!=y:
                    y1=floor(y)
                    y2=ceil(y)
                    
                else:
                    if y==0:
                        y1=0
                        y2=1
                    else:
                        y1=y-1
                        y2=y
                        
                
                
                # convert to int
                x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
                
                # we will apply bilinear equation
                # V = XA or A = inv(X)V
                
                X = np.array([[x1,y1,x1*y1,1],
                              [x1,y2,x1*y2,1],
                              [x2,y1,x2*y1,1],
                              [x2,y2,x2*y2,1]])
                
                V = np.array([[image[x1][y1]], [image[x1][y2]], [image[x2][y1]], [image[x2][y2]]])
                
                A = np.dot(np.linalg.inv(X),V)
                
                # now calculate V = XA
                
                final_image[i][j] = np.dot(np.array([x,y,x*y,1]),A)
                
    
    # mirrorizing boundaries (replacing all -1s with adjacent non -1 values)
    
    for i in range(M2):
        for j in range(Ny+1,N2):
            final_image[i][j]=final_image[i][j-1]
        
    for j in range(N2):
        for i in range(Mx+1,M2):
            final_image[i][j]=final_image[i-1][j]
            
    return final_image


# In[21]:


path = "/home/hardeekh/Desktop/IIIT/Semester 5 (Monsoon 2021)/DIP/A-1/"
name="x5.bmp"
image = cv2.imread(path+name,0)
c=0.125


# In[22]:


# this cell takes some time to execute
final_image = bilinear_interpolation(image,c)
cv2.imwrite("y5.bmp",final_image)
interpolated_image = cv2.imread(path+"y5.bmp",0)


# In[23]:


cv2.imshow("before interpolation", image)
cv2.imshow("after interpolation", interpolated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Q4 - Geometric Transformation of given Image

# In[89]:


import numpy as np, cv2
from math import *

## storing O to I mapping for Q5
O = []
I = []

def transform(image, T):
    
    
    M1, N1 = image.shape
    M2, N2 = M1,N1
    
    # initialize output matrix, -1 serves as a dummy value
    final_image = np.full((M2,N2),-1)
    
    midX, midY = image.shape
    #print(midX, midY)
    midX = midX//7
    midY = midY//7
    originX, originY = int(3*midX), int(3*midY)
    #print(midX,midY)
    
    T_inv = np.linalg.inv(T)
    #print(T_inv)
    count=0
    # iterating over values in output matrix corresponding to input matrix
    
    for i in range(M2):
        for j in range(N2):
            
            if final_image[i][j]==-1:
                
                # apply transformation
                
                i-=originX
                j-=originY
                input_matrix = np.dot(np.array([i,j,1]),T_inv)
                i+=originX
                j+=originY
                
                # find 4 nearest neighbours corresponding to input matrix
                x,y = input_matrix[0]+originX, input_matrix[1]+originY
                
                #if i==128 and j==128:
                 #   print("output:",i,j,"input:",x,y)
                    
                if x<originX or y<originY or x>=originX+midX or y>=originY+midY:
                    continue
                
                #print("output:",i,j,"input:",x,y)
                # x1<x2 and y1<y2
                
                if ceil(x)!=x and ceil(x)<originX+midX:
                    x1=floor(x)
                    x2=ceil(x)
                    
                else:
                    if x==0:
                        x1=0
                        x2=1
                    else:
                        x1=x-1
                        x2=x
                
            
                if ceil(y)!=y and ceil(y)<originY+midY:
                    y1=floor(y)
                    y2=ceil(y)
                    
                else:
                    if y==0:
                        y1=0
                        y2=1
                    else:
                        y1=y-1
                        y2=y
                        
                # convert to int
                x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
                
                if x1<originX or x2<originX or y1<originY or y2<originY or x1>=originX+midX or x2>=originX+midX or y1>=originY+midY or y2>=originY+midY:
                    continue
                # we will apply bilinear equation
                # V = XA or A = inv(X)V
                
                X = np.array([[x1,y1,x1*y1,1],
                              [x1,y2,x1*y2,1],
                              [x2,y1,x2*y1,1],
                              [x2,y2,x2*y2,1]])
                
                V = np.array([[image[x1][y1]], [image[x1][y2]], [image[x2][y1]], [image[x2][y2]]])
                
                A = np.dot(np.linalg.inv(X),V)
                
                
                # now calculate V = XA
                
                final_image[i][j] = np.dot(np.array([x,y,x*y,1]),A)
                
                O.append([int(i-originX),int(j-originY),1])
                I.append([int(x-originX),int(y-originY),1])
                
                #count
                count+=1
                
    print(count)
    return final_image


# In[90]:


import math

## default rotation is anti-clockwise
def get_rotation_matrix(degrees):
    
    s = math.sin(math.radians(degrees))
    c = math.sin(math.radians(degrees))
    
    T = np.array([[c,-s,0],
                  [s,c,0],
                  [0,0,1]])
    return T
    
def get_scaling_matrix(scale_x, scale_y):
    
    T = np.array([[scale_x,0,0],
                  [0,scale_y,0],
                  [0,0,1]])
    return T

## dis_x moves up/down and dis_y moves left/right
def get_translation_matrix(dis_x,dis_y):
    
    T = np.array([[1,0,0],
                  [0,1,0],
                  [dis_x,dis_y,1]])
    return T


# In[91]:


path = "/home/hardeekh/Desktop/IIIT/Semester 5 (Monsoon 2021)/DIP/A-1/"
name="img.bmp"
image = cv2.imread(path+name,0)
M,N = image.shape
padded_img = cv2.copyMakeBorder(image,int(3*M),int(3*M),int(3*N),int(3*N),cv2.BORDER_CONSTANT,None,value=0)

# create matrices
Ro = get_rotation_matrix(45) # 45 degrees clockwise
Sc = get_scaling_matrix(2,2)
Tr = get_translation_matrix(30,30)

T = np.dot(Ro,np.dot(Sc,Tr))

## note -> output image origin is shifted for convenient viewing and to ensure that image fits in screen
## negative coordinates are handled by shifting the origin for output image 


# In[92]:


T


# In[93]:


# this cell takes time to execute
import time
t = time.time()
transformed_img = transform(padded_img,T)
cv2.imwrite("y6.bmp",transformed_img)
transformed_image = cv2.imread(path+"y6.bmp",0)
print(time.time()-t)


# In[86]:


cv2.imshow("before transform", padded_img)
cv2.imshow("after transform", transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Q5 Image Registration

# In[123]:


O = np.array(O)
I = np.array(I)

## taking 20 points
O1 = O[:20,:]
I1 = I[:20,:]

O1, I1


# In[124]:


T1 = np.dot(np.dot(np.linalg.inv(np.dot(I1.T,I1)),I1.T),O1)

print(np.linalg.inv(T1))
T,T1


# In[125]:


import numpy as np, cv2
from math import *

def trans(image, T):
    
    
    M1, N1 = image.shape
    M2, N2 = M1,N1
    
    midX,midY = image.shape
    originX,originY=int(3*midX//7), int(3*midY//7)
    # initialize output matrix, -1 serves as a dummy value
    final_image = np.full((M2,N2),-1)
    
    ##print(T)
    T_inv = T#np.linalg.inv(T)
    t_inv = np.linalg.inv(T)
    #print(T_inv)
    count=0
    # iterating over values in output matrix corresponding to input matrix
    
    for i in range(M2):
        for j in range(N2):
            
            if final_image[i][j]==-1:
                
                # apply transformation
                
                i-=originX
                j-=originY
                input_matrix = np.dot(np.array([i,j,1]),T_inv)
                i+=originX
                j+=originY
                # find 4 nearest neighbours corresponding to input matrix
                x,y = input_matrix[0]+originX, input_matrix[1]+originY
                
                #if i==128 and j==128:
                 #   print("output:",i,j,"input:",x,y)
                    
                if x<0 or y<0 or x>=M2 or y>N2:
                    continue
                
                #print("output:",i,j,"input:",x,y)
                # x1<x2 and y1<y2
                
                if ceil(x)!=x and ceil(x)<M2:
                    x1=floor(x)
                    x2=ceil(x)
                    
                else:
                    if x==0:
                        x1=0
                        x2=1
                    else:
                        x1=x-1
                        x2=x
                
            
                if ceil(y)!=y and ceil(y)<N2:
                    y1=floor(y)
                    y2=ceil(y)
                    
                else:
                    if y==0:
                        y1=0
                        y2=1
                    else:
                        y1=y-1
                        y2=y
                        
                # convert to int
                x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
                
                if x1<0 or x2<0 or y1<0 or y2<0 or x1>=M2 or x2>=M2 or y1>=N2 or y2>=N2:
                    continue
                # we will apply bilinear equation
                # V = XA or A = inv(X)V
                
                X = np.array([[x1,y1,x1*y1,1],
                              [x1,y2,x1*y2,1],
                              [x2,y1,x2*y1,1],
                              [x2,y2,x2*y2,1]])
                
                V = np.array([[image[x1][y1]], [image[x1][y2]], [image[x2][y1]], [image[x2][y2]]])
                
                A = np.dot(np.linalg.inv(X),V)
                
                
                # now calculate V = XA
                
                final_image[i][j] = np.dot(np.array([x,y,x*y,1]),A)
                
                #count
                count+=1
                
    print(count)
    # mirrorizing boundaries (replacing all -1s with adjacent non -1 values)
    
    return final_image


# In[126]:


import time

t1 = time.time()

reg_img = trans(transformed_img,T1)

print(time.time()-t1)


# In[127]:


cv2.imwrite("z9.bmp",reg_img)
reg_img = cv2.imread(path+"z9.bmp",0)

cv2.imshow("before interpolation",transformed_image)
cv2.imshow("after interpolation", reg_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




