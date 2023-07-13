import os
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

from functools import reduce
from segment_anything import sam_model_registry, SamPredictor

models = {
    'vit_b': './checkpoints/sam_vit_b_01ec64.pth',
    'vit_l': './checkpoints/sam_vit_l_0b3195.pth',
    'vit_h': './checkpoints/sam_vit_h_4b8939.pth'
}

image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/truck.jpg"),0,[]]
]


def plot_boxes(img, boxes):
    img_pil = Image.fromarray(np.uint8(img * 255)).convert('RGB')
    draw = ImageDraw.Draw(img_pil)
    for box in boxes:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
    return img_pil


def segment_one(img, mask_generator, seed=None):
    if seed is not None:
        np.random.seed(seed)
    masks = mask_generator.generate(img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    mask_all = np.ones((img.shape[0], img.shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[m == True, i] = color_mask[i]
    result = img / 255 * 0.3 + mask_all * 0.7
    return result, mask_all


def run_inference(device, model_type, input_x, selected_points):
    if isinstance(input_x, int):
        input_x = cv2.imread(image_examples[input_x][0])
        input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)

    # sam model
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(input_x)  # Process the image to produce an image embedding

    points, labels, bboxes = [], [], []
    pnts_tmp, lbls_tmp, bbxs_tmp = [], [], None

    min_pnts = len(selected_points)
    for pnt, lbl in selected_points:
        if lbl == 2:
            if bbxs_tmp is None:
                bbxs_tmp = pnt
            else:
                if len(bboxes) > 0:
                    min_pnts = min(len(lbls_tmp), min_pnts)

                points.append(pnts_tmp)
                labels.append(lbls_tmp)
                bboxes.append([*bbxs_tmp, *pnt])

                pnts_tmp, lbls_tmp, bbxs_tmp = [], [], None
        else:
            pnts_tmp.append(pnt)
            lbls_tmp.append(lbl)


    B = len(bboxes)
    points.append(pnts_tmp)
    labels.append(lbls_tmp)

    if len(bboxes) < 2:
        points = [reduce(lambda x,y: x+y, points)]
        labels = [reduce(lambda x,y: x+y, labels)]
        min_pnts = len(labels[0])
    else:
        min_pnts = min(len(lbls_tmp), min_pnts)
        points = [points[i][:min_pnts] for i in range(1,B+1)]
        labels = [labels[i][:min_pnts] for i in range(1,B+1)]

    if min_pnts == 0:
        points, labels = None, None
    else:
        points = torch.Tensor(points).to(device)
        labels = torch.Tensor(labels).int().to(device)
        points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])

    if B == 0:
        bboxes = None
    else:
        bboxes = torch.Tensor(bboxes).to(device)
        bboxes = predictor.transform.apply_boxes_torch(bboxes, input_x.shape[:2])
    
    masks, scores, logits = predictor.predict_torch(
        point_coords=points,
        point_labels=labels,
        boxes=bboxes,
        multimask_output=False,
    )

    masks = masks.cpu().detach().numpy()
    mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))    
    for ann in masks:
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_all[ann[0] == True, i] = color_mask[i]

    img = input_x / 255 * 0.3 + mask_all * 0.7

    return img, mask_all
