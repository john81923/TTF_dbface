import common
import losses
import augment
import numpy as np
import math
import logging as log
import eval_tool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.utils.data import Dataset, DataLoader
from dbface import DBFace, DBFace_mbv2
from glob import glob
from os.path import basename, splitext
from tqdm import tqdm
import cv2
import sys
sys.path.append('../../../../klpr/applications/')
from utils_wpodocr import *
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../../../../kneron_sw_model/')
from licenseplate_ocr.licenseplate_ocr_postprocess import postprocess_


def ocrgrayscaleCvt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = 3
    t2 = 10
    k = 20
    reverseimg = np.zeros(gray.shape, np.uint8)+255

    corner_mean = np.mean(gray, axis=1)#img[0:t, 0:t]+ img[64-t:64, 0:t]+img[64-t:64, 160-t:160]+img[0:t, 160-t:160]
    corner_mean = (corner_mean[t:t2]+corner_mean[64-t2:64-t])/2
    corner_mean = np.mean(corner_mean)/255

    center_mean = np.mean(gray, axis=1)#img[ cth:64-cth , ctw:160-ctw]
    center_mean = center_mean[k:64-k]
    center_mean = np.mean(center_mean)/255

    #print('corner_mean:',corner_mean,', center_mean:',center_mean)
    flag =False
    if center_mean>corner_mean:
        img = reverseimg-gray
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        flag = True
    return img


class LDataset(Dataset):
    def __init__(self, labelfile, imagesdir, mean, std, width=800, height=800, classes=35):

        self.width = width
        self.height = height
        self.items = common.load_webface(labelfile, imagesdir)
        self.mean = mean
        self.std = std
        self.classes = classes
        self.vis_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/Lp_OCR_labels/ocr_vis_t'
        remove_folder([ self.vis_dir])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgfile, objs = self.items[index]

        image = common.imread(imgfile)
        #image = ocrgrayscaleCvt(image)

        classnum = self.classes
        # cvt

        if image is None:
            log.info("{} is empty, index={}".format(imgfile, index))
            return self[random.randint(0, len(self.items)-1)]

        keepsize            = 25
        image, objs = augment.webface(image, objs, self.width, self.height, keepsize=20)



        # grayscale
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # norm -0.5 ~ 0.5
        image = ((image / 256.0 - self.mean) / self.std).astype(np.float32)

        posweight_radius    = 2
        stride              = 4
        fm_width            = self.width // stride
        fm_height           = self.height // stride

        heatmap_gt          = np.zeros((classnum,     fm_height, fm_width), np.float32)
        heatmap_posweight   = np.zeros((classnum,     fm_height, fm_width), np.float32)
        keep_mask           = np.ones((classnum,     fm_height, fm_width), np.float32)
        reg_tlrb            = np.zeros((  4,    fm_height, fm_width), np.float32)
        reg_mask            = np.zeros((  1,    fm_height, fm_width), np.float32) # mask to classnum*4
        distance_map        = np.zeros((classnum,     fm_height, fm_width), np.float32) + 1000
        landmark_gt         = np.zeros((classnum * 10,fm_height, fm_width), np.float32)
        landmark_mask       = np.zeros((classnum,     fm_height, fm_width), np.float32)


        for obj in objs:
            isSmallObj = obj.area < keepsize * keepsize
            if isSmallObj:
                cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
                keep_mask[0, cy, cx] = 0                # no heap map regression
                w, h = obj.width / stride, obj.height / stride

                x0 = int(common.clip_value(cx - w // 2, fm_width-1))
                y0 = int(common.clip_value(cy - h // 2, fm_height-1))
                x1 = int(common.clip_value(cx + w // 2, fm_width-1) + 1)
                y1 = int(common.clip_value(cy + h // 2, fm_height-1) + 1)
                if x1 - x0 > 0 and y1 - y0 > 0:
                    keep_mask[0, y0:y1, x0:x1] = 0   # no heap map regression in box area
                hassmall = True

        hassmall = False
        finimg = ((image+0.5)*256.).copy()
        finimg = np.uint8(finimg)
        #finimgmask = np.zeros([64,160,3])
        bname = basename(splitext(imgfile)[0])
        for obj in objs:
            classes = obj.getlabel
            cx, cy, cx_diff, cy_diff = obj.safe_scale_center_and_diff(1 / stride, fm_width, fm_height)
            reg_box = np.array(obj.box) / stride
            isSmallObj = obj.area < keepsize * keepsize


            if isSmallObj:
                if obj.area >= 15*15:
                    distance_map[classes, cy, cx] = 0
                    reg_tlrb[:4, cy, cx] = reg_box
                    reg_mask[0, cy, cx] = 1   # reg box size
                continue

            if False:
                visx, visy, visr, visb = obj.box
                plot_one_box( [visx,visy,visr,visb], finimg, label=str(classes) )

            #w, h = obj.width / stride, obj.height / stride
            #x0 = (common.clip_value(cx+cx_diff - w / 2, fm_width-1))
            #y0 = (common.clip_value(cy+cy_diff - h / 2, fm_height-1))
            #x1 = (common.clip_value(cx+cx_diff + w / 2, fm_width-1) + 1)
            #y1 = (common.clip_value(cy+cy_diff + h / 2, fm_height-1) + 1)
            visx, visy, visr, visb = obj.box
            x0 = int(visx/stride)
            y0 = int(visy/stride)
            x1 = int(visr/stride+1)
            y1 = int(visb/stride+1)

            # Training Vis
            if True:
                plot_one_box( [x0*stride, y0*stride, x1*stride, y1*stride], finimg, label=str(classes) )

            if x1 - x0 > 0 and y1 - y0 > 0:
                keep_mask[0, y0:y1, x0:x1] = 1

            w_radius, h_radius = common.truncate_radius(( obj.width, obj.height))
            gaussian_map = common.draw_truncate_gaussian( heatmap_gt[classes, :, :], (cx, cy), h_radius, w_radius)

            mxface = 80
            miface = 25
            mxline = max(obj.width, obj.height)
            gamma = (mxline - miface) / (mxface - miface) * 10
            gamma = min(max(0, gamma), 10) + 1
            common.draw_gaussian(heatmap_posweight[classes, :, :], (cx, cy), posweight_radius, k=gamma)
            # print draw_gaussian

            #heatmap_draw = np.expand_dims( heatmap_posweight[classes, :, :], axis=2)*255.
            #heatmap_draw = np.uint8(heatmap_draw)
            #heatmap_draw = cv2.applyColorMap(heatmap_draw, cv2.COLORMAP_JET)
            #heatmap_draw = cv2.resize(heatmap_draw, finimg.shape[::-1][1:3] )
            #finimgmask = finimgmask+heatmap_draw #cv2.addWeighted(heatmap_draw, 0.3, finimg, 0.5, 0)

            range_expand_x = math.ceil(w_radius)
            range_expand_y = math.ceil(h_radius)

            min_expand_size = 3
            range_expand_x = max(min_expand_size, range_expand_x)
            range_expand_y = max(min_expand_size, range_expand_y)

            icx, icy = cx, cy
            reg_landmark = None
            fill_threshold = 0.3

            if obj.haslandmark:
                reg_landmark = np.array(obj.x5y5_cat_landmark) / stride
                x5y5 = [cx]*5 + [cy]*5
                rvalue = (reg_landmark - x5y5)
                landmark_gt[0:10, cy, cx] = np.array(common.log(rvalue)) / 4
                landmark_mask[0, cy, cx] = 1

            if not obj.rotate:
                for cx in range(icx - range_expand_x, icx + range_expand_x + 1):
                    for cy in range(icy - range_expand_y, icy + range_expand_y + 1):
                        if cx < fm_width and cy < fm_height and cx >= 0 and cy >= 0:

                            my_gaussian_value = 0.9
                            gy, gx = cy - icy + range_expand_y, cx - icx + range_expand_x
                            if gy >= 0 and gy < gaussian_map.shape[0] and gx >= 0 and gx < gaussian_map.shape[1]:
                                my_gaussian_value = gaussian_map[gy, gx]

                            distance = math.sqrt((cx - icx)**2 + (cy - icy)**2)
                            if my_gaussian_value > fill_threshold or distance <= min_expand_size:
                                already_distance = distance_map[classes, cy, cx]
                                my_mix_distance = (1 - my_gaussian_value) * distance

                                if my_mix_distance > already_distance:
                                    continue

                                distance_map[classes, cy, cx] = my_mix_distance
                                reg_tlrb[:4, cy, cx] = reg_box
                                reg_mask[0, cy, cx] = 1

        #finimg = cv2.addWeighted( (finimgmask).astype(np.uint8), 0.5, finimg, 0.5, 0)
        if True:
            cv2.imwrite( f'{self.vis_dir}/{bname}.jpg', finimg )

        #cv2.imwrite( f'/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/accuracy_output/{basename(imgfile)[:-5]}.jpg', image*256 + 128)
        # if hassmall:
        #     common.imwrite("test_result/keep_mask.jpg", keep_mask[0]*255)
        #     common.imwrite("test_result/heatmap_gt.jpg", heatmap_gt[0]*255)
        #     common.imwrite("test_result/keep_ori.jpg", (image*self.std+self.mean)*255)
        return T.to_tensor(image), heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, len(objs), keep_mask


class VDataset(Dataset):
    def __init__(self, imagesdir, mean, std):
        self.Cleaned_Data =  list( LabelMe_Parser( imagesdir ))
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.Cleaned_Data)

    def __getitem__(self, index):
        label, points, wh, img_file = self.Cleaned_Data[index]
        image = common.imread(img_file)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = ((image / 256.0 - self.mean) / self.std).astype(np.float32)
        image = np.expand_dims(image, 0)
        return T.to_tensor(image), label

class Load_Test_Data(Dataset):
    def __init__(self, img_n_json_pair, plate_w_threshold=0 ):
        self.img_n_json_pair = img_n_json_pair#[:500]
        self.licesnse_plate_count = 0
        self.plate_w_threshold = plate_w_threshold
        print( '\ttotal length:',len(img_n_json_pair) )

    def __copy__(self):
            return type(self)(self.img_n_json_pair, self.plate_w_threshold)

    def __getitem__(self, index):
        imgpath, jsonpath = self.img_n_json_pair[index]
        bname = basename( splitext(jsonpath)[0])
        #assert isfile(imgpath), imgpath
        image =  cv2.imread(imgpath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = (image / 256.0 - 0.5).astype(np.float32)

        with open( jsonpath, 'r') as json_file:
            data = json.load(json_file)
            dshapes = data['shapes']
            lplabel = []
            lpbox = []
            h = data['imageHeight']
            w = data['imageWidth']
            w,h = image.shape[1::-1]

            for idx, obj in enumerate(dshapes):
                boxlb = Label(rec2point( obj['points'], (w,h)))
                if boxlb.wh[0]<self.plate_w_threshold:
                    continue
                lplabel.append( obj['label'])
                lpbox.append( boxlb )
                self.licesnse_plate_count += 1
        return image, lplabel, lpbox, bname, jsonpath, imgpath

    def __len__(self):
        return len(self.img_n_json_pair)

def get_image_path_pair(wpod_val_data, image_dir ):
    img_n_json_pair = []
    for jsonpath in wpod_val_data:
        bname = basename( splitext(jsonpath)[0])
        imgpath = os.path.join( image_dir , bname+'.jpg')
        img_n_json_pair.append( (imgpath, jsonpath ))
    return img_n_json_pair


class App(object):
    def __init__(self, labelfile, imagesdir, valdata=None, load_checkpoint=None):
        self.writer = SummaryWriter()
        self.width, self.height = 160, 64
        #self.mean = [0.53301286, 0.51442557, 0.49572121]
        #self.std = [0.1813421, 0.18460054, 0.18462695]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [1., 1., 1.]

        'FROM_US_TRAIN_DATA'
        '''
        first
        mean = [0.408, 0.447, 0.47]
        std = [0.289, 0.274, 0.278]

        gray
        mean [0.51079454 0.51079454 0.51079454]
        std [0.18018843 0.18018843 0.18018843]

        OCR_LP_trainv688

        mean [0.53301286 0.51442557 0.49572121]
        std [0.1813421  0.18460054 0.18462695]
        '''

        self.batch_size = 32
        self.lr = 1e-4
        self.classes = 35
        self.gpus = [0] #[0, 1, 2, 3]
        self.gpu_master = self.gpus[0]
        if mbv_flag=='mbv2':
            print("\tMBV2")
            self.model = DBFace_mbv2(has_landmark=False, wide=64, has_ext=True, upmode="UCBA", classes=self.classes)
        else:
            print("\tMBV3")
            self.model = DBFace(has_landmark=False, wide=64, has_ext=True, upmode="UCBA", classes=self.classes)

        if load_checkpoint:
            if isfile(load_checkpoint):
                log.info( f'Load Weight: {load_checkpoint}')
                self.model.load(load_checkpoint)
            else:
                log.info( f'Load Weight Not Found: {load_checkpoint}')
                sys.exit(1)

        else:
            log.info( f'Init_weights')
            self.model.init_weights()
        self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.cuda(device=self.gpu_master)
        self.model.train()

        self.focal_loss = losses.FocalLoss()
        self.giou_loss = losses.GIoULoss()
        self.landmark_loss = losses.WingLoss(w=2)
        self.train_dataset = LDataset(labelfile, imagesdir, mean=self.mean, std=self.std, width=self.width, height=self.height, classes=self.classes)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24)
        print( 'Training data amount ', len(self.train_dataset))
        if valdata:
            self.val_dataset = valdata
            #self.val_loader = DataLoader(dataset=self.val_dataset , batch_size=self.batch_size, shuffle=False, num_workers=12)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.per_epoch_batchs = len(self.train_loader)
        self.iter = 0


        Nameidfile_path = '/mnt/models/CarLicensePlates_Detection_Recognition/ocr/ocr-net-TW.names'
        with open(Nameidfile_path, 'r') as Namep:
            Namefile = Namep.readlines()
            Namefile = { n.replace('\n', '' ):i for i, n in enumerate(Namefile) }
            self.RevNamefile = { str(n[1]):n[0] for n in Namefile.items()}


    def set_lr(self, lr):
        self.lr = lr
        log.info(f"setting learning rate to: {lr}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


    def train_epoch(self, epoch):

        for indbatch, (images, heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, num_objs, keep_mask) in enumerate(self.train_loader):
            self.iter += 1

            batch_objs = sum(num_objs)
            batch_size = self.batch_size

            if batch_objs == 0:
                batch_objs = 1

            heatmap_gt          = heatmap_gt.to(self.gpu_master)
            heatmap_posweight   = heatmap_posweight.to(self.gpu_master)
            keep_mask           = keep_mask.to(self.gpu_master)
            reg_tlrb            = reg_tlrb.to(self.gpu_master)
            reg_mask            = reg_mask.to(self.gpu_master)
            landmark_gt         = landmark_gt.to(self.gpu_master)
            landmark_mask       = landmark_mask.to(self.gpu_master)
            images              = images.to(self.gpu_master)

            hm, tlrb  = self.model(images)
            hm = hm.sigmoid()
            hm = torch.clamp(hm, min=1e-4, max=1-1e-4)
            tlrb = torch.exp(tlrb)

            hm_loss = self.focal_loss(hm, heatmap_gt, heatmap_posweight, keep_mask=keep_mask) / batch_objs
            reg_loss = self.giou_loss(tlrb, reg_tlrb, reg_mask)*5
            #landmark_loss = self.landmark_loss(landmark, landmark_gt, landmark_mask)*0.1
            loss = hm_loss + reg_loss #+ landmark_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_flt = epoch + indbatch / self.per_epoch_batchs

            self.writer.add_scalar('Loss/train', loss.item(), self.iter)

            if indbatch % 500 == 0:
                #self.writer.add_scalar('Loss/train', hm_loss.item(), epoch)
                log.info(
                    f"iter: {self.iter}, lr: {self.lr:g}, epoch: {epoch_flt:.2f}, loss: {loss.item():.2f}, \n\thm_loss: {hm_loss.item():.2f}, "
                    f"box_loss: {reg_loss.item():.2f}"
                )

            #    log.info("save hm")
            #    hm_image = hm[0, 0].cpu().data.numpy()
            #    #common.imwrite(f"{jobdir}/imgs/hm_image.jpg", hm_image * 255)
            #    #common.imwrite(f"{jobdir}/imgs/hm_image_gt.jpg", heatmap_gt[0, 0].cpu().data.numpy() * 255)
            #
            #    image = np.clip((images[0].permute(1, 2, 0).cpu().data.numpy() * self.std + self.mean) * 255, 0, 255).astype(np.uint8)
            #    outobjs = eval_tool.detect_images_giou_with_netout(hm, tlrb, threshold=0.5, ibatch=0)
            #
            #    im1 = image.copy()
            #    #common.imwrite(f"{jobdir}/imgs/train_result_org.jpg", im1)
            #    for obj in outobjs:
            #        common.drawbbox(im1, obj, Namefile=RevNamefile)
            #    common.imwrite(f"{jobdir}/imgs/train_result_{self.iter}.jpg", im1)


    def val(self, epoch):
        self.common_mistakes = { 'D':'0','0':'13', 'G':'6','6':'16', 'B':'8','8':'11', 'Z':'7','7':'34', 'S':'9','9':'27' }
        self.common_mistakes_A20 = { 'D':['0'], 'G':'6', 'B':'8', 'Z':['7','2'], 'S':'9' }
        self.common_mistakes_02A = { '0':'13','2':'34','6':'16','8':'11','7':'34', '9':'27' }
        self.rule_dict4 = {'00AA':1,'000A':1,'00A0':1,'A000':1,'0A00':1,'AA00':1,'AAA0':1 }
        self.rule_dict5 = {'AA000':1,'A0000':1,'0A000':1,'AAA00':1,'000AA':2,'000A0':2,'0000A':2}
        self.rule_dict6 = {'AAA000':2,'A0A000':2,'0AA000':2,'000AAA':2,'AA0000':1,'A00000':1,'0000AA':3,'0000A0':3, '0A0000':1}
        self.rule_dict7 = {'AAA0000':2}
        rules = {
            "common_mistakes":self.common_mistakes,
            "common_mistakes_A20":self.common_mistakes_A20,
            "common_mistakes_02A":self.common_mistakes_02A,
            "rule_dict4":self.rule_dict4,
            "rule_dict5":self.rule_dict5,
            "rule_dict6":self.rule_dict6,
            "rule_dict7":self.rule_dict7
        }
        log.info(f'do evalutation {len(self.val_dataset)}')
        tp = ocr_count = ocr_tp = ocr_fn = ocr_fp = 0
        with torch.no_grad():
            for frame, lplabels, lpboxs, bname, jsonpath, imgpath in tqdm(self.val_dataset):
                for lpgt, lpbox in zip(lplabels, lpboxs):
                    lp_img = self.get_lp_img(frame, lpbox, (160,64))
                    model_output = self.model(lp_img)
                    predictions = self.predicton_process(model_output)
                    ocr_output = postprocess_( predictions , threshold=0.2, nms_iou=0.45, cls=0, RevNamefile=self.RevNamefile, rules=rules )
                    ocr_str = ocr_output[0] if ocr_output else ''
                    lpgt = lpgt.replace('-', '')
                    ocr_str = ocr_str.replace('-', '')
                    if ocr_str:
                        if ocr_str == lpgt:
                            ocr_tp += 1
                        else:
                            ocr_fp += 1
                        ocr_count += 1
                    else:
                        ocr_fn += 1
            log.info(f"epoch: {epoch} - top1_val:{ocr_tp/len(self.val_dataset):.2f} fp_val:{ocr_fp/len(self.val_dataset):.2f} ocr_count:{ocr_count/len(self.val_dataset):.2f} fn_val:{ocr_fn/len(self.val_dataset):.2f} ")

    def predicton_process(self, model_output):
        center, box = model_output
        center = center.sigmoid()
        box = torch.exp(box)
        print(box.shape)
        center = torch.clamp(center, min=1e-4, max=1-1e-4)
        hm_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding = 1)(center)
        return center.cpu().numpy(), box.cpu().numpy(), hm_pool.cpu().numpy()

    def get_lp_img(self, frame, lpbox ,wh):
        tl = np.amin( lpbox.pts_2d, axis=0).astype(np.int)
        br = np.amax( lpbox.pts_2d, axis=0 ).astype(np.int)

        frame_cp = cv2.resize( frame[tl[1]:br[1], tl[0]:br[0]] , wh)
        return torch.unsqueeze( T.to_tensor(frame_cp ), 0)


    def train(self):
        self.epochs = 600
        lr_scheduer = {
            2: 1e-3,
            4: 2e-3,
            6: 1e-3,
            400:1e-4,
            500:1e-5
        }

        # finetune
        finetune = False
        if finetune:
            lr_scheduer = {
                0: 1e-4,
                200: 1e-5
            }
            self.epochs = 300

        # train
        self.model.train()
        for epoch in range(self.epochs):



            if epoch in lr_scheduer:
                self.set_lr(lr_scheduer[epoch])

            self.train_epoch(epoch)
            if (epoch) %100 ==0 and epoch >= 100:
                file = f"{jobdir}/ocr_{epoch}.pth"
                common.mkdirs_from_file_path(file)
                torch.save(self.model.module.state_dict(), file)
                log.info(f'Saved Model:{file}')

            elif epoch == self.epochs-1:
                file = f"{jobdir}/ocr_final.pth"
                common.mkdirs_from_file_path(file)
                torch.save(self.model.module.state_dict(), file)
                log.info(f'Saved Model:{file}')

            if (epoch)%50==0:
                log.info('Validation')
                self.val(epoch)


mbv_flag = 'mbv2'
trial_name = f"030321_{mbv_flag}_license_plate_dbocr_tw_test"
#trial_name = f"120720_{mbv_flag}_license_plate_dbocr_tw_AddCutmix_mxf80_smrt"

print( 'Train Model', trial_name)
jobdir = f"/mnt/models/CarLicensePlates_Detection_Recognition/ocr/testing/{trial_name}"
log.basicConfig(level=log.DEBUG)

base_path = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_'
image_dir = os.path.join( base_path, 'labelme_0730-0807')
image_dir2 = os.path.join( base_path, 'lane_data_1019')
#wpod_dir = os.path.join(base_path, 'Lp_OCR_labels')
wpod_val_data = glob( f'{base_path}/Lp_OCR_labels/val_0730-0807/*.json')
wpod_val_data2 = glob( f'{base_path}/Lp_OCR_labels/val_1019/*.json')

img_n_json_pair = get_image_path_pair(wpod_val_data , image_dir)
img_n_json_pair += get_image_path_pair(wpod_val_data2 , image_dir2)

#output_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/hw_accuracy_output_onnx'
plate_w_threshold = 0
training_data = Load_Test_Data( img_n_json_pair[:100] , plate_w_threshold=plate_w_threshold )

#
#train_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/Lp_OCR_labels/ocr_train_1204_3k'
#train_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/Lp_OCR_labels/ocr_train_1207_infce_clean_7k'
train_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/Lp_OCR_labels/ocr_train_1229_cutmix_clean_allmix'
#train_dir = '/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/Lp_OCR_labels/ocr_train_1007/valid'

print( 'Training Data', basename(train_dir))

app = App( f'{train_dir}/label.txt', f'{train_dir}/images', valdata=training_data )#,
    #load_checkpoint='/mnt/models/CarLicensePlates_Detection_Recognition/ocr/testing/122320_mbv3_license_plate_dbocr_tw_1207_scratch_ep600/ocr_300.pth')
    #load_checkpoint='/mnt/models/CarLicensePlates_Detection_Recognition/ocr/released/102720_license_plate_dbocr_tw/ocr_mbv3_model_1008.pth')
            #f'/mnt/atdata/CarLicensePlates_Detection_Recognition/CF_Lane_/labelme_0730-0807' )
app.train()
