import os
import cv2
import numpy as np
import gradio as gr
from inference import run_inference


# points color and marker
colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255)]
markers = [5, 5, 0]

# image examples
# in each list, the first element is image path,
# the second is id (used for original_image State),
# the third is an empty list (used for selected_points State)
image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/truck.jpg"),0,[]]
]


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Segment Anything Labeler
            Extended from: https://github.com/5663015/segment_anything_webui
            '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(["vit_b", "vit_l", "vit_h"], value='vit_b', label="Select Model")
            # select device
            device = gr.Dropdown(["cpu", "cuda"], value='cpu', label="Select Device")

    # Segment image
    with gr.Tab(label='Labeling'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    with gr.Row():
                        gr.Markdown('Default: bounding box point.\n\n'
                                    'Bounding box must be: x1 < x2, y1 < y2.\n\n'
                                    'For multiple bboxes, fore-background points associate with preceding bbox. '
                                    'Each bbox will have an equal number of points associated to it (takes first K available).')
                        undo_button = gr.Button('Undo point')
                    radio = gr.Radio(['foreground point', 'background point', 'bounding box point'], label='point labels')

                # run button
                button = gr.Button("Auto!")
            # show the image with mask
            with gr.Tab(label='Image+Mask'):
                output_image = gr.Image(type='numpy')
            # show only mask
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='numpy')
        def process_example(img, ori_img, sel_p):
            return ori_img, []

        example = gr.Examples(
            examples=image_examples,
            inputs=[input_image, original_image, selected_points],
            outputs=[original_image, selected_points],
	        fn=process_example,
	        run_on_click=True
        )

    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return img, []  # when new image is uploaded, `selected_points` should be empty

    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    def draw_img(img, sel_pix):
        last_bbox = None
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
            if label == 2:
                if last_bbox is None:
                    last_bbox = point
                else:
                    x1, y1 = last_bbox
                    x2, y2 = point
                    assert x1 < x2 and y1 < y2

                    cv2.rectangle(img, (x1, y1), (x2, y2), colors[2], 2)
                    last_bbox = None

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground point':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background point':
            sel_pix.append((evt.index, 0))    # append the background_point
        elif point_type == 'bounding box point':
            sel_pix.append((evt.index, 2))
        else:
            sel_pix.append((evt.index, 2))    # default bbox point

        # draw points
        draw_img(img, sel_pix)
        
        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)

    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()

        if len(sel_pix) != 0:
            sel_pix.pop()
            # draw points
            draw_img(temp, sel_pix)

        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)

    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image]
    )

    # button image
    button.click(run_inference, inputs=[device, model_type, original_image, selected_points],
                 outputs=[output_image, output_mask])

demo.queue().launch(debug=True, enable_queue=True)



