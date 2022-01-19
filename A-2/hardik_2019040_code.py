#!/usr/bin/env python
# coding: utf-8

# # DIP A-2 2019040

# ## Q3 Histogram Equalization

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def norm_hist(in_img):
    
    m,n = in_img.shape
    
    return np.bincount(in_img.flatten())/(m*n)

def get_cdf(norm_hist):
    
    return np.array([sum(norm_hist[:i+1]) for i in range(len(norm_hist))])

def hist_eq(in_img):
    
    in_h = norm_hist(in_img)
    in_H = get_cdf(in_h)
    
    out_img = np.array(in_img, copy=True)
    
    s = np.array(255*in_H, dtype=int)
    
    L = len(in_H)
    
    for i in range(L):
        out_img[in_img==i]=s[i]
        
    out_h = norm_hist(out_img)
    out_H = get_cdf(out_h)
    
    return in_img, in_h, in_H, out_img, out_h, out_H

def plot_hist(hist, title):
    
    plt.title(title)
    plt.plot(hist, color='r')
    plt.bar(np.arange(len(hist)), hist, color='r')
    plt.xlabel("Pixel values (0-255)")
    plt.ylabel("Fraction of Pixels")
    plt.show()
    
def plot_cdf(cdf, title):
    
    plt.title(title)
    plt.plot(cdf, color='b')
    plt.xlabel("Pixel values (0-255)")
    plt.ylabel("Fraction of Pixels")
    plt.show()
    


# In[63]:


path = "/home/hardeekh/Desktop/IIIT/Semester 5 (Monsoon 2021)/DIP/A-2/"
name="x5.bmp"
img = cv2.imread(path+name,0)

in_img, in_h, in_H, out_img, out_h, out_H = hist_eq(img)


# In[64]:


## plotting
plot_hist(in_h,"Input Image normalized histogram")
plot_hist(out_h,"Equalized Image normalized histogram")
plot_cdf(in_H,"Input Image CDF")
plot_cdf(out_H,"Equalized Image CDF")


# In[65]:


## display images

cv2.imwrite("in_eq.bmp",in_img)
cv2.imwrite("out_eq.bmp",out_img)

i=cv2.imread(path+"in_eq.bmp",0)
e=cv2.imread(path+"out_eq.bmp",0)

cv2.imshow("Input",i)
cv2.imshow("Equalized",e)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Q4 Histogram Matching

# In[66]:


def hist_matching(I, gamma):
    
    M = np.array(I,copy=True)
    
    T = np.array(255*np.power(I/255,gamma),dtype=int)
    
    h = norm_hist(I)
    g = norm_hist(T)
    H = get_cdf(h)
    G = get_cdf(g)
    
    L = len(H)
    
    for i in range(L):
        
        if H[i]==0:
            continue
            
        argmin=None
        mindiff=None
        
        for j in range(len(G)):
            
            if G[j]==0:
                continue
                
            if mindiff is None or abs(H[i]-G[j]) < mindiff:
                argmin=j
                mindiff=abs(H[i]-G[j])
        
        
        M[I==i]=argmin
        
    m = norm_hist(M)
    
    return I, h, T, g, M, m
                
        


# In[67]:


path = "/home/hardeekh/Desktop/IIIT/Semester 5 (Monsoon 2021)/DIP/A-2/"
name="x5.bmp"
img = cv2.imread(path+name,0)
I,h,T,g,M,m = hist_matching(img,0.5)


# In[68]:


## plotting

plot_hist(h,"Input Image normalized histogram")
plot_hist(g,"Target Image normalized histogram")
plot_hist(m,"Matched Image normalized histogram")
# plot_cdf(in_H,"Input Image CDF")
# plot_cdf(out_H,"Equalized Image CDF")


# In[69]:


## display images

cv2.imwrite("I.bmp",I)
cv2.imwrite("T.bmp",T)
cv2.imwrite("M.bmp",M)

i=cv2.imread(path+"I.bmp",0)
t=cv2.imread(path+"T.bmp",0)
m=cv2.imread(path+"M.bmp",0)

cv2.imshow("Input",i)
cv2.imshow("Target",t)
cv2.imshow("Matched",m)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Q5 Convolution

# In[13]:


##### generate two random 3x3 matrices

def rotate_filter(f):
    
    return np.flip(np.flip(f,axis=0),axis=1)
      
def conv(I,f):
    
    n=I.shape[0]
    m=f.shape[0]
    size=n+m-1
    
    O = np.zeros((size,size))
    
    I_pad = np.pad(I,(m-1,m-1),constant_values=0)
    
    f_rot = rotate_filter(f)
    
    for i in range(m//2,size+m//2):
        for j in range(m//2,size+m//2):
            
            O[i-(m//2),j-(m//2)]= np.sum(I_pad[i-(m//2):i+(m//2+1),j-(m//2):j+(m//2+1)]*f_rot)
    
    print("Input:\n",I,"\n")
    print("Filter:\n",f,"\n")
    print("Rotated Filter:\n",f_rot,"\n")
    print("Output:\n",O,"\n")
    
    return O

n=3
m=3
## generate 3x3 random matrices of type int with range 0-255
I = np.random.randint(low=0,high=256,size=(n,n))
f = np.random.randint(low=0,high=256,size=(m,m))
O=conv(I,f)


# In[ ]:




