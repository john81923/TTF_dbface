

import numpy as np
import common
import torch.nn as nn
import torch
import json
import cv2

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def detect_images_giou_with_netout(output_hm, output_tlrb, output_landmark=None, threshold=0.4, ibatch=0 ):

    stride = 4
    _, num_classes, hm_height, hm_width = output_hm.shape
    #print(num_classes)
    hm = output_hm[ibatch].reshape(1, num_classes, hm_height, hm_width)
    box_classes = 1
    tlrb = output_tlrb[ibatch].cpu().data.numpy().reshape(1, box_classes * 4, hm_height, hm_width)

    nmskey = _nms(hm, 5)
    kscore, kinds, kcls, kys, kxs = _topk(nmskey, 15)
    kys = kys.cpu().data.numpy().astype(np.int)
    kxs = kxs.cpu().data.numpy().astype(np.int)
    kcls = kcls.cpu().data.numpy().astype(np.int)

    key = [[], [], [], []]
    for ind in range(kscore.shape[1]):
        score = kscore[0, ind]
        if score > threshold:
            key[0].append(kys[0, ind])
            key[1].append(kxs[0, ind])
            key[2].append(score)
            key[3].append(kcls[0, ind])

    imboxs = []
    if key[0] is not None and len(key[0]) > 0:
        ky, kx = key[0], key[1]
        classes = key[3]

        scores = key[2]

        for i in range(len(kx)):
            class_ = classes[i]

            cx, cy = kx[i], ky[i]
            x1, y1, x2, y2 = tlrb[0, :4, cy, cx]
            x1, y1, x2, y2 = (np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])) * stride


            imboxs.append(common.BBox(label=str(class_), xyrb=common.floatv([x1, y1, x2, y2]), score=scores[i].item()))
    return imboxs


def detect_images_giou_with_retinaface_style_eval(output_hm, output_tlrb, output_landmark, threshold=0.4, ibatch=0):

    stride = 4
    _, _, hm_height, hm_width = output_hm.shape
    hm = output_hm[ibatch].reshape(1, 1, hm_height, hm_width)
    tlrb = output_tlrb[ibatch]
    landmark = output_landmark[ibatch]

    area = hm_height * hm_width
    keep = (hm > threshold).view(area)
    indices = torch.arange(0, area)[keep]
    hm = hm.view(1, area).cpu().data.numpy()
    tlrb = tlrb.view(4, area).cpu().data.numpy()
    landmark = landmark.view(10, area).cpu().data.numpy()

    cx, cy = indices % hm_width, indices // hm_width
    scores = hm[0, indices]
    x1, y1, x2, y2 = tlrb[0:4, indices]
    cts = np.vstack([cx, cy, cx, cy])
    locs = np.vstack([-x1, -y1, x2, y2])
    x1, y1, x2, y2 = (cts + locs) * stride

    x5y5 = landmark[0:10, indices]
    x5y5 = common.exp(x5y5 * 4)
    x5y5 = (x5y5 + np.vstack([cx] * 5 + [cy] * 5)) * stride

    imboxs = []
    for i in range(len(indices)):
        boxlandmark = list(zip(x5y5[0:5, i], x5y5[5:, i]))
        imboxs.append(common.BBox(label="facial", xyrb=common.floatv([x1[i], y1[i], x2[i], y2[i]]), score=scores[i], landmark=boxlandmark))
    return imboxs


def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)

    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append( obj )
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep

def dbocr_postprocess( hm ,box ,threshold, nms_iou, RevNamefile):
    stride = 4
    hm_pool = _nms(hm, 3)
    btsize = hm.shape[0]
    #print('hm_shape :',hm_pool.shape)
    kscore, kinds, kcls, kys, kxs = _topk(hm_pool, 100)
    #print(hm_pool.shape)
    #print(box.shape)
    kys = kys.cpu().data.numpy().astype(np.int)
    kxs = kxs.cpu().data.numpy().astype(np.int)
    kcls = kcls.cpu().data.numpy().astype(np.int)
    box = box.cpu().data.numpy()

    val_result = []
    for btid in range(btsize):

        key = [[], [], [], []]
        for ind in range(kscore.shape[1]):
            score = kscore[btid, ind]
            if score > threshold:
                key[0].append(kys[btid, ind])
                key[1].append(kxs[btid, ind])
                key[2].append(score)
                key[3].append(kcls[btid, ind])

        imboxs = []
        lp_top2 = ''

        if key[0] is not None and len(key[0]) > 0:
            ky, kx = key[0], key[1]
            classes = key[3]
            scores = key[2]

            for i in range(len(kx)):
                class_ = classes[i]

                cx, cy = kx[i], ky[i]
                x1, y1, x2, y2 = box[ btid, :, cy, cx]
                x1, y1, x2, y2 = (np.array([cx, cy, cx, cy]) + np.array([-x1, -y1, x2, y2])) * stride

                imboxs.append(common.BBox(label=str(class_), xyrb=common.floatv([x1, y1, x2, y2]), score=scores[i].item()))
                imboxs = nms(imboxs, iou=nms_iou)
                imboxs.sort( key=lambda x: x.center[0] )

            for obj in imboxs:
                lp_top2 += RevNamefile[obj.label]

            val_result.append(lp_top2)

    return  val_result



def detect_image(model, image, mean, std, threshold=0.4):
    image = common.pad(image)
    image = ((image / 255 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    center, box, landmark = model(image)

    center = center.sigmoid()
    box = torch.exp(box)
    return detect_images_giou_with_netout(center, box, landmark, threshold)


def detect_image_retinaface_style(model, image, mean, std, threshold=0.4):
    image = common.pad(image)
    image = ((image / 255 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    center, box, landmark = model(image)

    center = center.sigmoid()
    box = torch.exp(box)
    return detect_images_giou_with_retinaface_style_eval(center, box, landmark, threshold)
