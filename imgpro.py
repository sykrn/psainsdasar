# instructor used only
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# all these function can be imported by student
clip_matrix_value = np.clip


def clip(a,min=None,max=None):
    return np.clip(a,min,max)
    
def open_image(img_path,size=False):
    img = Image.open(img_path)
    if isinstance(size,tuple): img = img.resize(size, Image.ANTIALIAS)
    return img

def get_RGB(image):
    img = np.asanyarray(image)
    print('Image dimmension is',img.shape[:2],'with',img.shape[-1],'channels' )
    return img[:,:,0],img[:,:,1],img[:,:,2]

def show_img(imgs):
    plt.figure()
    if isinstance(imgs,(list,tuple)): 
        imgs = np.dstack(imgs)
        plt.imshow(imgs)
    else:
        plt.imshow(imgs,cmap='gray')

def generate_tmatrix(img):
    seed = img.sum()
    h,w = img.shape    
    np.random.seed(int(seed))
    
    t = np.eye(3)
    t[:2,-1] = [-w/2,-h/2]

    m = np.zeros((3,3))
    m[:2,:2] = np.random.randint(0,20,size=(2,2))/20
    m+= np.eye(3)*0.5
    m[-1,-1] = 1

    m = m@t
    t[:2,-1] = [w/2,h/2]
    m = t@m

    return m
