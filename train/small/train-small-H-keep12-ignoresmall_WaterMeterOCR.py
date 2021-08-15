
import common
import losses
import augment_wm_ocr
import numpy as np
import math
import logger
import eval_tool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.utils.data import Dataset, DataLoader
from dbface import DBFace, DBFace_mbv2
from glob import glob
from os.path import basename
from tqdm import tqdm

class LDataset(Dataset):
    def __init__(self, labelfile, imagesdir, mean, std, width=800, height=800, classes=35):

        self.width = width
        self.height = height
        self.items = common.load_webface(labelfile, imagesdir)
        self.mean = mean
        self.std = std
        self.classes = classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgfile, objs = self.items[index]
        #    #    WaterMeter Needs to Ignore the Last Number!  #
        image = common.imread(imgfile)
        classnum = self.classes

        if image is None:
            log.info("{} is empty, index={}".format(imgfile, index))
            return self[random.randint(0, len(self.items)-1)]

        keepsize            = 7
        image, objs = augment_wm_ocr.webface(image, objs, self.width, self.height, keepsize=0)

        # norm
        image = ((image / 255.0 - self.mean) / self.std).astype(np.float32)

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




        hassmall = False
        for obj in objs:
            isSmallObj = obj.area < keepsize * keepsize

            if isSmallObj:
                print('s')
                target_classID = obj.getlabel
                cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
                keep_mask[target_classID, cy, cx] = 0
                w, h = obj.width / stride, obj.height / stride

                x0 = int(common.clip_value(cx - w // 2, fm_width-1))
                y0 = int(common.clip_value(cy - h // 2, fm_height-1))
                x1 = int(common.clip_value(cx + w // 2, fm_width-1) + 1)
                y1 = int(common.clip_value(cy + h // 2, fm_height-1) + 1)
                if x1 - x0 > 0 and y1 - y0 > 0:
                    keep_mask[target_classID, y0:y1, x0:x1] = 0
                hassmall = True

        for obj in objs:
            classes = obj.getlabel
            cx, cy = obj.safe_scale_center(1 / stride, fm_width, fm_height)
            reg_box = np.array(obj.box) / stride
            isSmallObj = obj.area < keepsize * keepsize

            if isSmallObj:
                if obj.area >= 5 * 5:
                    distance_map[classes, cy, cx] = 0
                    reg_tlrb[:4, cy, cx] = reg_box
                    reg_mask[0, cy, cx] = 1
                continue

            w, h = obj.width / stride, obj.height / stride
            x0 = int(common.clip_value(cx - w // 2, fm_width-1))
            y0 = int(common.clip_value(cy - h // 2, fm_height-1))
            x1 = int(common.clip_value(cx + w // 2, fm_width-1) + 1)
            y1 = int(common.clip_value(cy + h // 2, fm_height-1) + 1)
            if x1 - x0 > 0 and y1 - y0 > 0:
                keep_mask[classes, y0:y1, x0:x1] = 1

            w_radius, h_radius = common.truncate_radius((obj.width, obj.height))
            gaussian_map = common.draw_truncate_gaussian(heatmap_gt[classes, :, :], (cx, cy), h_radius, w_radius)

            mxface = 25
            miface = 5
            mxline = max(obj.width, obj.height)
            gamma = (mxline - miface) / (mxface - miface) * 10
            gamma = min(max(0, gamma), 10) + 1
            common.draw_gaussian(heatmap_posweight[classes, :, :], (cx, cy), posweight_radius, k=gamma)

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
        # if hassmall:
        #     common.imwrite("test_result/keep_mask.jpg", keep_mask[0]*255)
        #     common.imwrite("test_result/heatmap_gt.jpg", heatmap_gt[0]*255)
        #     common.imwrite("test_result/keep_ori.jpg", (image*self.std+self.mean)*255)
        return T.to_tensor(image), heatmap_gt, heatmap_posweight, reg_tlrb, reg_mask, landmark_gt, landmark_mask, len(objs), keep_mask




class VDataset(Dataset):
    def __init__(self, imagesdir, mean, std):
        self.items = glob('%s/*.png'%imagesdir)[:640]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        imgfile = self.items[index]
        image = common.imread(imgfile)
        image = ((image / 255.0 - self.mean) / self.std).astype(np.float32)
        return T.to_tensor(image), basename(imgfile).split('_')[1]


class Set_sizeWH:
    def __init__(self, wh):
        self._w = wh[0]
        self._h = wh[1]

    @property
    def h(self):
        return self._h
    @property
    def w(self):
        return self._w


class App(object):
    def __init__(self, labelfile, imagesdir, valimgdir, set_size):

        self.width, self.height = set_size.w, set_size.h
        self.mean = [0.53301286, 0.51442557, 0.49572121]
        self.std = [0.1813421, 0.18460054, 0.18462695]

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

        self.batch_size = 64
        self.lr = 1e-4
        self.classes = 10
        self.gpus = [0] #[0, 1, 2, 3]
        self.gpu_master = self.gpus[0]
        #self.model = DBFace(has_landmark=False, wide=64, has_ext=True, upmode="UCBA", classes=self.classes)
        self.model = DBFace_mbv2(has_landmark=False, wide=64, has_ext=True, upmode="UCBA", classes=self.classes)

        self.model.init_weights()
        self.model = nn.DataParallel(self.model, device_ids=self.gpus)
        self.model.cuda(device=self.gpu_master)
        self.model.train()

        self.focal_loss = losses.FocalLoss()
        self.giou_loss = losses.GIoULoss()
        self.landmark_loss = losses.WingLoss(w=2)
        self.train_dataset = LDataset(labelfile, imagesdir, mean=self.mean, std=self.std, width=self.width, height=self.height, classes=self.classes)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24)

        self.val_dataset = VDataset(valimgdir,  mean=self.mean, std=self.std )
        self.val_loader = DataLoader(dataset=self.val_dataset , batch_size=self.batch_size, shuffle=False, num_workers=12)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.per_epoch_batchs = len(self.train_loader)
        self.iter = 0
        self.epochs = 155

        Nameidfile_path = '/mnt/atdata/WaterMeter/ocr-wm.names'
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

            if indbatch % 2000 == 0:
                log.info(
                    f"iter: {self.iter}, lr: {self.lr:g}, epoch: {epoch_flt:.2f}, loss: {loss.item():.2f}, \n\thm_loss: {hm_loss.item():.2f}, "
                    f"box_loss: {reg_loss.item():.2f}"
                )
            #    log.info("save hm")
            #    hm_image = hm[0, 0].cpu().data.numpy()
            #    #common.imwrite(f"{jobdir}/imgs/hm_image.jpg", hm_image * 255)
            #    #common.imwrite(f"{jobdir}/imgs/hm_image_gt.jpg", heatmap_gt[0, 0].cpu().data.numpy() * 255)
            #
                image = np.clip((images[0].permute(1, 2, 0).cpu().data.numpy() * self.std + self.mean) * 255, 0, 255).astype(np.uint8)
                outobjs = eval_tool.detect_images_giou_with_netout(hm, tlrb, threshold=0.5, ibatch=0)
            #
                im1 = image.copy()
            #    #common.imwrite(f"{jobdir}/imgs/train_result_org.jpg", im1)
                for obj in outobjs:
                    common.drawbbox(im1, obj, Namefile=self.RevNamefile)
                common.imwrite(f"{jobdir}/imgs/train_result_{self.iter}.jpg", im1)


    def val(self, epoch):
        log.info(f'do evalutation {len(self.val_dataset)}')
        tp = 0
        with torch.no_grad():
            ocr_count = ocr_top1 = ocr_fn = ocr_fp = 0
            for imgval, valgt_lp in tqdm(self.val_loader):
                hm_v, tlrb_v  = self.model(imgval)
                hm_v = hm_v.sigmoid()
                hm_v = torch.clamp(hm_v, min=1e-4, max=1-1e-4)
                tlrb_v = torch.exp(tlrb_v)

                val_result = eval_tool.dbocr_postprocess(  hm_v ,tlrb_v, threshold=0.5, nms_iou=0.45, RevNamefile=self.RevNamefile )

                for lp_top2, gt_lp in zip(val_result, valgt_lp):
                    if lp_top2:
                        if gt_lp == lp_top2:
                            ocr_top1 += 1
                        elif len(gt_lp)!= len(lp_top2):
                            ocr_fp += 1
                        ocr_count += 1
                    else:
                        ocr_fn += 1
            log.info(f"epoch: {epoch} - top1_val:{ocr_top1/len(self.val_dataset):.2f} fp_val:{ocr_fp/len(self.val_dataset):.2f} ocr_count:{ocr_count/len(self.val_dataset):.2f} fn_val:{ocr_fn/len(self.val_dataset):.2f} ")


    def train(self):

        lr_scheduer = {
            1: 1e-3,
            2: 2e-3,
            3: 1e-3,
            60: 1e-4,
            120: 1e-5
        }

        # train
        self.model.train()
        for epoch in range(self.epochs):
            if epoch in lr_scheduer:
                self.set_lr(lr_scheduer[epoch])

            self.train_epoch(epoch)
            if epoch %50 ==0:
                file = f"{jobdir}/models/{epoch + 1}.pth"
                common.mkdirs_from_file_path(file)
                torch.save(self.model.module.state_dict(), file)
            #if epoch %20 == 0:
            #    self.val(epoch)


trial_name = "small-UCBA-mbv2-WaterMeterOCR_rotate12_scale08"
jobdir = f"jobs/{trial_name}"

log = logger.create(trial_name, f"{jobdir}/logs/{trial_name}.log")
#app = App("webface/train/label.txt", "webface/WIDER_train/images")
#app = App("/mnt/sdd/craig/face_detection/webface/train/label.txt", "/mnt/sdd/craig/face_detection/webface/WIDER_train/images")
app = App('/mnt/atdata/WaterMeter/OcrWM_96_32/train/label.txt',
            '/mnt/atdata/WaterMeter/OcrWM_96_32/train/images',
            '/mnt/atdata/WaterMeter/OcrWM_96_32/train/images',
            set_size=Set_sizeWH( (96, 32) ) )
app.train()
