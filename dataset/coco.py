import os
import os.path as osp
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torch.utils import data
import random
import glob
import math
import cv2
import json

import imgaug as ia
import imgaug.augmenters as iaa
import math

class sampled_aug(object):
    def __init__(self):
        self.affinity = iaa.Affine(rotate=(-30, 30),shear=(-20, 20),scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        self.ela = iaa.ElasticTransformation(alpha = 50, sigma = 5)
    def __call__(self,image,label):
        image,label = self.affinity(image = image,segmentation_maps = label[np.newaxis,:,:,np.newaxis])
        # image,label = self.ela(image = image,segmentation_maps = label)   
        label = label[0,:,:,0]
        return image,label

class Coco_MO_Train(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, img_dir,json_path):
        self.image_dir = img_dir


        self.K = 11
        self.skip = 0
        with open(json_path) as f:
            data = json.load(f)

        self.images_list = data['images']
        self.anno_list = data['annotations']
        self.sampled_aug = sampled_aug()
        #self.augment = PhotometricDistort()

    def __len__(self):
        return len(self.images_list)

    def change_skip(self,f):
        self.skip = f

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def Augmentation(self, image, label,sampled_f_m = None):
        # Scaling
        h,w = label.shape
        if w<h:
            factor = 480/w
            image = cv2.resize(image, (480, int(factor * h)), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (480, int(factor * h)), interpolation=cv2.INTER_NEAREST)             
        else:
            factor = 480/h
            image = cv2.resize(image, (int(factor * w), 480), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (int(factor * w), 480), interpolation=cv2.INTER_NEAREST)           

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

        h,w = label.shape

        #affinity
        image1 = image.copy()
        label1 = label.copy()
        image2 = image.copy()
        label2 = label.copy()

        dst_points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
        tx1 = random.randint(-w//10,w//10)
        ty1 = random.randint(-h//10,h//10)
        tx2 = random.randint(-w//10,w//10)
        ty2 = random.randint(-h//10,h//10)
        tx3 = random.randint(-w//10,w//10)
        ty3 = random.randint(-h//10,h//10)
        tx4 = random.randint(-w//10,w//10)
        ty4 = random.randint(-h//10,h//10)
        src_points = np.float32([[0 + tx1,0 + ty1],[0 + tx2,h-1 + ty2],[w-1 + tx3,h-1 + ty3],[w-1 + tx4,0 + ty4]])
        H1,_ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)

        dst_points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
        tx1_ = random.randint(0,w//10)
        ty1_ = random.randint(0,h//10)
        tx2_ = random.randint(0,w//10)
        ty2_ = random.randint(0,h//10)
        tx3_ = random.randint(0,w//10)
        ty3_ = random.randint(0,h//10)
        tx4_ = random.randint(0,w//10)
        ty4_ = random.randint(0,h//10)
        src_points = np.float32([[0 + tx1 + tx1_ * tx1/(abs(tx1)+1e-5),0 + ty1 + ty1_ * ty1/(abs(ty1)+1e-5)],[0 + tx2 + tx2_ * tx2/(abs(tx2)+1e-5),h-1 + ty2 + ty2_ * ty2/(abs(ty2)+1e-5)],[w-1 + tx3 + tx3_ * tx3/(abs(tx3)+1e-5),h-1 + ty3 + ty3_ * ty3/(abs(ty3)+1e-5)],[w-1 + tx4 + tx4_ * tx4/(abs(tx4)+1e-5),0 + ty4 + ty4_ * ty4/(abs(ty4)+1e-5)]])
        H2,_ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)

        image1 = cv2.warpPerspective(image1, H1, (w,h),flags = cv2.INTER_LINEAR) 
        label1 = cv2.warpPerspective(label1, H1, (w,h),flags = cv2.INTER_NEAREST)   
        image2 = cv2.warpPerspective(image2, H2, (w,h),flags = cv2.INTER_LINEAR) 
        label2 = cv2.warpPerspective(label2, H2, (w,h),flags = cv2.INTER_NEAREST)      

        ob_loc = ((label + label1 + label2) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        if x_max - x_min >384:
            start_w = random.randint(x_min,x_max - 384)
        elif x_max - x_min == 384:
            start_w = x_min
        else:
            start_w = random.randint(max(0,x_max-384), min(x_min,w - 384))

        if y_max - y_min >384:
            start_h = random.randint(y_min,y_max - 384)
        elif y_max - y_min == 384:
            start_h = y_min
        else:
            start_h = random.randint(max(0,y_max-384), min(y_min,h - 384))
        # Cropping

        end_h = start_h + 384
        end_w = start_w + 384


        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        image1 = image1[start_h:end_h, start_w:end_w]
        label1 = label1[start_h:end_h, start_w:end_w]
        image2 = image2[start_h:end_h, start_w:end_w]
        label2 = label2[start_h:end_h, start_w:end_w]

        if sampled_f_m != None:
            for sf,sm in sampled_f_m:
                h,w = sm.shape
                frame1x = random.randint(0,383)
                frame1y = random.randint(0,383)
                frame2x = min(383,max(0,random.randint(0,40) + frame1x))
                frame2y = min(383,max(0,random.randint(0,20) + frame1y))
                frame3x = min(383,max(0,random.randint(0,40) + frame2x))
                frame3y = min(383,max(0,random.randint(0,20) + frame2y))

                sf_c = sf.copy()
                sf_c[sm == 0] = 0
                label[max(0,frame1y-h//2):min(384,frame1y + h - h//2),max(0,frame1x - w//2):min(384,frame1x + w - w//2)][sm[max(0,h//2 - frame1y):min(h,h//2+384-frame1y),max(0,w//2-frame1x):min(w,w//2+384-frame1x)] != 0] = np.max(sm)
                image[max(0,frame1y-h//2):min(384,frame1y + h - h//2),max(0,frame1x - w//2):min(384,frame1x + w - w//2)][sm[max(0,h//2 - frame1y):min(h,h//2+384-frame1y),max(0,w//2-frame1x):min(w,w//2+384-frame1x)] != 0] = 0
                image[max(0,frame1y-h//2):min(384,frame1y + h - h//2),max(0,frame1x - w//2):min(384,frame1x + w - w//2)] = image[max(0,frame1y-h//2):min(384,frame1y + h - h//2),max(0,frame1x - w//2):min(384,frame1x + w - w//2)] + sf_c[max(0,h//2 - frame1y):min(h,h//2+384-frame1y),max(0,w//2-frame1x):min(w,w//2+384-frame1x)]

                sf,sm = self.sampled_aug(sf,sm)
                sf_c = sf.copy()
                sf_c[sm == 0] = 0
                label1[max(0,frame2y-h//2):min(384,frame2y + h - h//2),max(0,frame2x - w//2):min(384,frame2x + w - w//2)][sm[max(0,h//2 - frame2y):min(h,h//2+384-frame2y),max(0,w//2-frame2x):min(w,w//2+384-frame2x)] != 0] = np.max(sm)
                image1[max(0,frame2y-h//2):min(384,frame2y + h - h//2),max(0,frame2x - w//2):min(384,frame2x + w - w//2)][sm[max(0,h//2 - frame2y):min(h,h//2+384-frame2y),max(0,w//2-frame2x):min(w,w//2+384-frame2x)] != 0] = 0
                image1[max(0,frame2y-h//2):min(384,frame2y + h - h//2),max(0,frame2x - w//2):min(384,frame2x + w - w//2)] = image1[max(0,frame2y-h//2):min(384,frame2y + h - h//2),max(0,frame2x - w//2):min(384,frame2x + w - w//2)] + sf_c[max(0,h//2 - frame2y):min(h,h//2+384-frame2y),max(0,w//2-frame2x):min(w,w//2+384-frame2x)]

                sf,sm = self.sampled_aug(sf,sm)
                sf_c = sf.copy()
                sf_c[sm == 0] = 0
                label2[max(0,frame3y-h//2):min(384,frame3y + h - h//2),max(0,frame3x - w//2):min(384,frame3x + w - w//2)][sm[max(0,h//2 - frame3y):min(h,h//2+384-frame3y),max(0,w//2-frame3x):min(w,w//2+384-frame3x)] != 0] = np.max(sm)
                image2[max(0,frame3y-h//2):min(384,frame3y + h - h//2),max(0,frame3x - w//2):min(384,frame3x + w - w//2)][sm[max(0,h//2 - frame3y):min(h,h//2+384-frame3y),max(0,w//2-frame3x):min(w,w//2+384-frame3x)] != 0] = 0
                image2[max(0,frame3y-h//2):min(384,frame3y + h - h//2),max(0,frame3x - w//2):min(384,frame3x + w - w//2)] = image2[max(0,frame3y-h//2):min(384,frame3y + h - h//2),max(0,frame3x - w//2):min(384,frame3x + w - w//2)] + sf_c[max(0,h//2 - frame3y):min(h,h//2+384-frame3y),max(0,w//2-frame3x):min(w,w//2+384-frame3x)]


        image = image /255.
        image1 = image1 /255.
        image2 = image2 /255.


        return [image,image1,image2], [label,label1,label2]

    def mask_process(self,mask,f,num_object,ob_list):
        n = num_object
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        if f == 0:
            for i in range(1,11):
                if np.sum(mask == i) > 350:
                    n += 1
                    ob_list.append(i)
            if n > 5:
                n = 5
                ob_list = random.sample(ob_list,n)
        for i,l in enumerate(ob_list):
            mask_[mask == l] = i + 1
        return mask_,n,ob_list 

    def __getitem__(self, index):
        images = self.images_list[index]
        image_name = images['file_name']
        url = images['coco_url']
        id_ = images['id']
        instances_list = []
        for anno in self.anno_list:
            if anno['image_id'] == id_:
                instances_list.append(anno)

        info = {}
        info['name'] = image_name
        info['num_frames'] = 3

        N_frames = np.empty((3,)+(384,384,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        frames_ = []
        masks_ = []

        # print(os.path.join(self.image_dir,image + '.jpg'),os.path.join(self.mask_dir,image + '.png'))
        frame = np.array(Image.open(os.path.join(self.image_dir,image_name)).convert('RGB'))
        h,w,_ = frame.shape
        mask = np.zeros((h,w,20)).astype(np.uint8)

        if random.random() < 0.5:
            object_index = 1
            mask_list = []
            for inst in instances_list:
                segs = inst['segmentation']
                segs_list = []
                try:
                    for seg in segs:
                        if len(np.array(seg).shape) == 0:
                            continue
                        tmp = np.array(seg).reshape(-1,2).astype(np.int32)
                        segs_list.append(tmp)
                    tmp_mask = np.zeros((h,w))
                    tmp_mask = cv2.fillPoly(tmp_mask, segs_list,1)
                    if np.sum(tmp_mask) < 2000:
                        continue
                    mask_list.append(tmp_mask)
                    object_index += 1   
                except:
                    pass


           	


            for i,tmp_mask in enumerate(mask_list):
                mask[:,:,i+1] = tmp_mask

            mask = np.argmax(mask,axis = 2).astype(np.uint8)

            if len(mask_list) != 0:
                frames_,masks_ = self.Augmentation(frame,mask)
            else:
                tmp_sample_num = random.randint(1,3)
                sampled_objects = []
                max_iter = 20
                while tmp_sample_num > 0 and max_iter > 0:
                    max_iter -= 1
                    tmp = random.sample(self.anno_list,1)
                    if tmp[0]['bbox'][2] * tmp[0]['bbox'][3] < 3000:
                        continue
                    sampled_objects.append(tmp[0])
                    tmp_sample_num -= 1
            

            

                sampled_f_m = []
                for sampled_object in sampled_objects:
                    ob_img_path = os.path.join(self.image_dir,str(sampled_object['image_id']).zfill(12) + '.jpg')
                    ob_frame = np.array(Image.open(ob_img_path).convert('RGB'))
                    h,w,_ = ob_frame.shape
                    ob_segs = sampled_object['segmentation']
                    ob_bbox = sampled_object['bbox']
                    ob_segs_list = []
                    try:
                        for ob_seg in ob_segs:
                            tmp = np.array(ob_seg).reshape(-1,2).astype(np.int32)
                            ob_segs_list.append(tmp)
                        ob_mask = np.zeros((h,w)).astype(np.uint8)
                        ob_mask = cv2.fillPoly(ob_mask, ob_segs_list,object_index)
                        object_index += 1

                        y1,y2 = int(ob_bbox[1]),int(ob_bbox[1] + ob_bbox[3])
                        x1,x2 = int(ob_bbox[0]),int(ob_bbox[0] + ob_bbox[2])
                        ob_mask = ob_mask[y1:y2,x1:x2]
                        ob_frame = ob_frame[y1:y2,x1:x2,:]

                        save_h_w = int(math.sqrt(ob_bbox[3] ** 2 + ob_bbox[2] ** 2))

                        ob_mask = np.lib.pad(ob_mask,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),(int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2))),'constant',constant_values=0)
                        ob_frame = np.lib.pad(ob_frame,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),(int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2)),(0,0)),'constant',constant_values=0)

                        # cv2.imwrite('test.jpg',ob_frame)
                        # cv2.imwrite('test.png',ob_mask*255)
                        sampled_f_m.append([ob_frame,ob_mask])
                    except:
                        pass        
                frames_,masks_ = self.Augmentation(frame,mask,sampled_f_m)

        else:

            object_index = 1
            mask_list = []
            for inst in instances_list:
                segs = inst['segmentation']
                segs_list = []
                try:
                    for seg in segs:
                        if len(np.array(seg).shape) == 0:
                            continue
                        tmp = np.array(seg).reshape(-1,2).astype(np.int32)
                        segs_list.append(tmp)
                    tmp_mask = np.zeros((h,w))
                    tmp_mask = cv2.fillPoly(tmp_mask, segs_list,1)
                    if np.sum(tmp_mask) < 2000:
                        continue
                    mask_list.append(tmp_mask)
                    object_index += 1   
                except:
                    pass
            if len(mask_list) > 5:
                mask_list = random.sample(mask_list,5)

            object_index = len(mask_list) + 1

            for i,tmp_mask in enumerate(mask_list):
                mask[:,:,i+1] = tmp_mask

            mask = np.argmax(mask,axis = 2).astype(np.uint8)

            tmp_sample_num = random.randint(1,3)
            sampled_objects = []
            max_iter = 20
            while tmp_sample_num > 0 and max_iter > 0:
                max_iter -= 1
                tmp = random.sample(self.anno_list,1)
                if tmp[0]['bbox'][2] * tmp[0]['bbox'][3] < 3000:
                    continue
                sampled_objects.append(tmp[0])
                tmp_sample_num -= 1
            

            

            sampled_f_m = []
            for sampled_object in sampled_objects:
                ob_img_path = os.path.join(self.image_dir,str(sampled_object['image_id']).zfill(12) + '.jpg')
                ob_frame = np.array(Image.open(ob_img_path).convert('RGB'))
                h,w,_ = ob_frame.shape
                ob_segs = sampled_object['segmentation']
                ob_bbox = sampled_object['bbox']
                ob_segs_list = []
                try:
                    for ob_seg in ob_segs:
                        tmp = np.array(ob_seg).reshape(-1,2).astype(np.int32)
                        ob_segs_list.append(tmp)
                    ob_mask = np.zeros((h,w)).astype(np.uint8)
                    ob_mask = cv2.fillPoly(ob_mask, ob_segs_list,object_index)
                    object_index += 1

                    y1,y2 = int(ob_bbox[1]),int(ob_bbox[1] + ob_bbox[3])
                    x1,x2 = int(ob_bbox[0]),int(ob_bbox[0] + ob_bbox[2])
                    ob_mask = ob_mask[y1:y2,x1:x2]
                    ob_frame = ob_frame[y1:y2,x1:x2,:]

                    save_h_w = int(math.sqrt(ob_bbox[3] ** 2 + ob_bbox[2] ** 2))

                    ob_mask = np.lib.pad(ob_mask,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),(int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2))),'constant',constant_values=0)
                    ob_frame = np.lib.pad(ob_frame,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),(int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2)),(0,0)),'constant',constant_values=0)

                    # cv2.imwrite('test.jpg',ob_frame)
                    # cv2.imwrite('test.png',ob_mask*255)
                    sampled_f_m.append([ob_frame,ob_mask])
                except:
                    pass        
            frames_,masks_ = self.Augmentation(frame,mask,sampled_f_m)

        num_object = 0
        ob_list = []
        for f in range(3):
            tmp_mask,num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            N_frames[f],N_masks[f] = frames_[f],tmp_mask

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        if num_object == 0:
            num_object += 1
        num_objects = torch.LongTensor([num_object])
        return Fs, Ms, num_objects, info


if __name__ == '__main__':
    import os
    import sys
    pwd = os.getcwd()
    sys.path.append(pwd)
    from utils.helpers import overlay_davis
    import matplotlib.pyplot as plt
    import pdb
    import argparse
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-o", type=str, help="", default='./tmp')
        parser.add_argument("-Dcoco", type=str, help="path to coco",default='/smart/haochen/cvpr/data/COCO/coco/')
        parser.add_argument("-Ddavis", type=str, help="path to davis",default='/smart/haochen/cvpr/data/DAVIS/')
        return parser.parse_args()
    args = get_arguments()
    davis_root = args.Ddavis
    coco_root = args.Dcoco
    output_dir = args.o
    dataset = Coco_MO_Train('{}train2017'.format(coco_root),'{}annotations/instances_train2017.json'.format(coco_root))
    palette = Image.open('{}Annotations/480p/blackswan/00000.png'.format(davis_root)).getpalette()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,(Fs,Ms,num_objects,info) in enumerate(dataset):
        mask = np.argmax(Ms.numpy(), axis=0).astype(np.uint8)
        img_list = []
        for f in range(3):
            pF = (Fs[:,f].permute(1,2,0).numpy()*255.).astype(np.uint8)
            pE = mask[f]
            canvas = overlay_davis(pF, pE, palette)
            img = np.concatenate([pF,canvas],axis = 0)
            img_list.append(img)
        out_img = np.concatenate(img_list,axis = 1)
        out_img = Image.fromarray(out_img)
        print('saving images to {}'.format(output_dir))
        out_img.save(os.path.join(output_dir, str(i).zfill(5) + '.jpg'))
        pdb.set_trace()
