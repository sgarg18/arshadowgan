import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from networks import *
import numpy as np
from preprocess import *
from pathlib import Path

# from .postprocess import *

class Spinny_Shadow:
    def __init__(self):
        self.preprocess=Preprocess()
        # self.postprocess=Postprocess()
        self.encoder='resnet18'
        self.generator=Generator_with_Refin(self.encoder)
        self.generator_path ="/datadrive1/automobiles/checkpoints_aluminium_woref/SG_generator17_2928.pth"
        # self.generator_path ="/datadrive1/automobiles/checkpoints_final/SG_generator9_2668.pth"

        self.device='cpu'

    def load_model(self):
        checkpoint = torch.load(self.generator_path,map_location='cpu')
        # checkpoint = torch.load(self.generator_path)
        self.generator.load_state_dict(checkpoint['state_dict'])
        self.generator.to(self.device)
        self.generator.eval()
        print("checkpoints loaded")

    def get_inference(self,input_image,noshadow_image):
        with torch.no_grad():
            shadow_mask_tensor1,shadow_mask_tensor2=self.generator(input_image)
            result = torch.add(noshadow_image, shadow_mask_tensor2)
            m_result=torch.min(result)
            result=(result-m_result)/(torch.max(result)-m_result)

        output_image = np.uint8(255 * (result.detach().cpu().numpy()[0].transpose(1,2,0)))    
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        return output_image

    def process_image(self,img_url,flag):
        self.load_model()
        if flag==False:
            output_dir='output_img17_2928'
            os.makedirs(output_dir, exist_ok=True)
            images_list = list(Path(img_url).rglob("*.[pPjJ][nNpP][gG]"))
            for image_path in images_list:
                image_path = str(image_path)
                head, tail = os.path.split(image_path)

                input_image,transparent_image,noshadow_image=self.preprocess.main(image_path,flag)
                output=self.get_inference(input_image,noshadow_image)
                cv2.imwrite(os.path.join(output_dir, tail),output)
        else:
            input_image,transparent_image,noshadow_image=self.preprocess.main(img_url,flag)
            output=self.get_inference(input_image,noshadow_image)
            cv2.imwrite('output_img/bad_car_26_1499.jpg',output)

        # final_output=self.postprocess.main(output,transparent_image)
        return output


file='https://storage.googleapis.com/spyne/AI/app/edited/remove_bg__3b3cccf6-b321-46b8-950c-382e5febfadb.png'
file="https://storage.googleapis.com/spyne/AI/app/edited/remove_bg__de1979b2-6f27-4546-a5c2-030fae25352c.png"
# file="https://storage.googleapis.com/spyne/AI/app/edited/remove_bg__ed415724-974b-4e08-9ad7-47ebe807bf54.png"
aa=Spinny_Shadow()
flag=False
file='/datadrive1/automobiles/data/bad_car_rmbg'
out=aa.process_image(file,flag)








# bg_path='/datadrive1/automobiles/data/bg_wc_cleanup.jpg'
# img=cv2.imread(bg_path)    
# img=cv2.resize(img,(1920,1080),interpolation = cv2.INTER_CUBIC)
# cv2.imwrite('output_img/bad_car_26_1499.jpg',out)


