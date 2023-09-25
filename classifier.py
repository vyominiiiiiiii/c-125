import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
X_train_data,X_test_data,y_train_data,y_test_data=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
X_train_data_scaled=X_train_data/255.0
X_test_data_scaled=X_test_data/255.0

clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_data_scaled,y_train_data)
def getPrediction(img1):
    im_pil=Image.open(img1)
    img_bw=im_pil.convert("L")
    img_bw_resize=img_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(img_bw_resize,pixel_filter)
    img_bw_resize_inverted_scaled=np.clip(img_bw_resize-min_pixel,0,255)
    max_pixel=np.max(img_bw_resize)
    img_bw_resize_inverted_scaled=np.asarray(img_bw_resize_inverted_scaled)/max_pixel

    test_sample=np.array(img_bw_resize_inverted_scaled).reshape(1,784)
    test_pred=clf.predict(test_sample)
    return test_pred[0]
