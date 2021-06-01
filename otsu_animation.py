import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import *

def dullrazor(img, lowbound=15, showimgs=True, filterstruc=3, inpaintmat=3):
    #grayscale
    imgtmp1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #applying a blackhat
    filterSize =(filterstruc, filterstruc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize) 
    imgtmp2 = cv2.morphologyEx(imgtmp1, cv2.MORPH_BLACKHAT, kernel)

    #0=skin and 255=hair
    ret, mask = cv2.threshold(imgtmp2, lowbound, 255, cv2.THRESH_BINARY)
    
    #inpainting
    img_final = cv2.inpaint(img, mask, inpaintmat ,cv2.INPAINT_TELEA)
    
    if showimgs:
        print("_____DULLRAZOR_____")
        plt.imshow(imgtmp1, cmap="gray")
        plt.show()
        plt.imshow(imgtmp2, cmap='gray')
        plt.show()
        plt.imshow(mask, cmap='gray')
        plt.show()
        plt.imshow(img_final)
        plt.show()
        print("___________________")

    return img_final

def otsu(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(img,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    list_x=list()
    list_within_variance=list()
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
        list_x.append(i)
        list_within_variance.append(fn)
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    print( "{} {}".format(thresh,ret) )

    # plt.plot(hist_norm)
    # plt.show()
    # plt.plot(list_x,list_within_variance)
    # plt.show()

    return ret, hist_norm, list_x,list_within_variance

if __name__ == '__main__':
    img=cv2.imread("input_image.jpeg")
    img2=dullrazor(img,showimgs=False)
    # img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret,hist_norm,list_x,list_within_variance=otsu(img2)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(list_x,list_within_variance)
    vl=ax1.axvline(list_x[0],color='r',linewidth=2)

    temp_list_x=list()
    temp_thresh_img=list()

    for i in tqdm(range(0,256)):
        # blur = cv2.GaussianBlur(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),(5,5),0)
        temp_ret,temp_th=cv2.threshold(img2,i,255,cv2.THRESH_BINARY_INV)
        temp_list_x.append(i)
        cv2.imwrite("temp_image.jpg",temp_th)
        temp_thresh_img.append(cv2.imread("temp_image.jpg",0))
        # temp_thresh_img.append(temp_th)

    cutoff_img=ax2.imshow(temp_thresh_img[0],'gray')

    temp_dict=dict(zip(list_x,list_within_variance))
    
    text_desc=ax2.text(0,1400,"threshold={0:d}, value={1:.2f}".format(0,temp_dict.get(0,np.nan)))
    ax1.title.set_text("Within class variance plot")
    ax2.title.set_text("Mask from otsu method")

    def animate(i,vl,cutoff_img,text_desc):
        vl.set_xdata([i,i])
        text_desc.set_text("threshold={0:d}, value={1:.2f}".format(i,temp_dict.get(i,np.nan)))
        # temp_ret,temp_th=cv2.threshold(img2,i,255,cv2.THRESH_BINARY)
        # ax2.clear()
        # cutoff_img=ax2.imshow(temp_thresh_img[i],'gray')
        # ax2.set_color('gray')
        cutoff_img.set_data(temp_thresh_img[i])
        return vl, cutoff_img,text_desc

    ani = animation.FuncAnimation(
        fig, animate, frames=temp_list_x, interval=10, fargs=(vl,cutoff_img,text_desc)
    )
    # plt.show()
    print("saving image")
    writer=animation.PillowWriter(fps=30)
    ani.save("otsu_threshold.gif",writer=writer)
    print("image saved")