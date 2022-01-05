import os 
import cv2 
from PIL import Image, ImageOps
import numpy as np
import requests
from PIL import Image
import PIL
import io
import torch

class Preprocess():
    def __init__(self):
        self.device='cpu'
        self.bg_path='/datadrive1/automobiles/data/bg_wc_cleanup.jpg'

    def download_image(self, url):
        resp = requests.get(url)
        resp.raise_for_status()
        return PIL.Image.open(io.BytesIO(resp.content))


    def load_image(self,noshadow_image,mask):
        w,h=1024,576
        noshadow_image=cv2.resize(noshadow_image,(w,h),interpolation = cv2.INTER_CUBIC)
        mask=cv2.resize(mask,(w,h),interpolation = cv2.INTER_CUBIC)
        mask[mask >= 128] = 255; mask[mask < 128] = 0
        image1, mask1 = noshadow_image.astype(np.float) / 127.5 - 1.0, \
        								mask.astype(np.float) / 127.5 - 1.0

        mask1=torch.from_numpy(mask1)
        image1=torch.from_numpy(image1)
        image1=torch.transpose(image1,0,2)
        image1=torch.transpose(image1,1,2)
        mask = torch.unsqueeze(mask1, 0).to(device=self.device, dtype=torch.float)
        mask = torch.unsqueeze(mask, 1).to(device=self.device, dtype=torch.float)
        noshadow_image = torch.unsqueeze(image1,0).to(device=self.device, dtype=torch.float)
        input_image = torch.cat((noshadow_image, mask), axis=1)
        return input_image,noshadow_image


    def rescale(self,transparent_image,fine_height=1080, fine_width=1920):
        roi_factor=0.54
        transparent_arr=np.array(transparent_image,dtype=np.uint8)

        mask = transparent_arr[:, :, 3]
        image_h, image_w = mask.shape

        ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)
        points = cv2.findNonZero(mask)  
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(points)
        x, y, w, h = roi_x, roi_y, roi_w, roi_h
        
        ROI = transparent_arr[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w, :]
        ROI = transparent_arr[y : y + h, x : x + w, :]

        
        new_roi_h = fine_height * roi_factor
        new_roi_w = roi_w * (fine_height / roi_h) * roi_factor

        new_roi_w, new_roi_h = int(new_roi_w), int(new_roi_h)

        margin = 0

        print("new_roi_w and new_roi_h", new_roi_w, new_roi_h)
        start_x = int((fine_width - new_roi_w) / 2)
        start_y = int((fine_height - new_roi_h) / 2) - margin
        print("start_x and start_y", start_x, start_y)

        ROI=cv2.resize(ROI,(new_roi_w, new_roi_h),interpolation = cv2.INTER_CUBIC)


        transparent_arr_scale = np.zeros((fine_height, fine_width, 4), dtype=np.uint8)
        transparent_arr_scale[
            start_y : start_y + new_roi_h,
            start_x : start_x + new_roi_w,
            :,
        ] = ROI

        return transparent_arr_scale


    def prepare_infer_images(self,transparent_image):
        transparent_image=Image.fromarray(transparent_image)
        # bg_img=Image.open(self.bg_path)
        bg_img=Image.new("RGB",(1920,1080),(157,154,149))
        bg_img=bg_img.resize((1920,1080))
        bg_img.paste(transparent_image,(0,0),transparent_image)

        input_image_raw=np.array(bg_img, dtype=np.uint8)
        
        _,_,_,mask=transparent_image.split()
        mask=np.array(mask, dtype=np.uint8)
        input_image,noshadow_image=self.load_image(input_image_raw,mask)

        return input_image,noshadow_image

    def main(self,img_url,url_flag=True):
        if url_flag==True:
            transparent_image=self.download_image(img_url)
        else:
            transparent_image=Image.open(img_url)
        transparent_arr=self.rescale(transparent_image)
        input_image,noshadow_image=self.prepare_infer_images(transparent_arr)
        return input_image,transparent_image,noshadow_image


    





