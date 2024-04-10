import os
import re
from PIL import Image, ImageDraw, ImageSequence, ImageFont
from moviepy.editor import ImageSequenceClip
import numpy as np
import torch
import gradio as gr
from diffusers import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_image import StableVideoDiffusionPipeline
import openai
from openai import OpenAI

client=OpenAI(api_key='<KEY>')
client.api_key='sk-YKep5XVdwxSsRmRR8c19013379374b66895785Ea60262147'
client.base_url='https://api.gptplus5.com/v1'

# load pipe
pretrained_model_path = "stabilityai/stable-video-diffusion-img2vid"
unet = UNetSpatioTemporalConditionModel.from_pretrained("/path/to/satge2/unet", torch_dtype=torch.float16,)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    pretrained_model_path, 
    unet=unet,
    torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True)
pipe = pipe.to('cuda:0')

def normalize_coordinates(arr, width, height):
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    x1_normalized = x1 / width
    y1_normalized = y1 / height
    x2_normalized = x2 / width
    y2_normalized = y2 / height

    normalized_arr = np.stack((x1_normalized, y1_normalized, x2_normalized, y2_normalized), axis=1)
    return normalized_arr

def text_to_array(text):
    # remove
    text = re.sub(r'[\[\]\s]', '', text) 
    # split
    numbers = text.split(',')
    # str2int
    numbers = [int(num) for num in numbers]
    array = np.array(numbers).reshape(-1, 4)
    return array

def export_to_video(frames, video_path):
    clip = ImageSequenceClip(list(frames), fps=7)
    # write video file
    clip.write_videofile(video_path, codec='libx264')

# YOLOv5 detection function
def detect(img, model):
    results = model(img).xyxy[0][:, :4]
    return results.tolist()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# TrackDiffusion")
    with gr.Row():
        img_input = gr.Image()
        img_output = gr.AnnotatedImage()
        video_output = gr.Video()

    detect_btn = gr.Button("Detect Objects")
    gen_btn = gr.Button("Gen Video")
    selected_object = gr.Textbox(label="Selected Object")
    text_prompt = gr.Textbox(label="Object Motion Description")

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    rec_list = []
    global chosen_obj

    def run_detection(img,):
        rec_list.clear()
        img_pil = Image.fromarray(img)
        sections = []
        _sections = detect(img_pil, model)
        ins_idx = 0
        for sec in _sections:
            x1, y1, x2, y2 = sec
            rec_list.append((int(x1), int(y1), int(x2), int(y2)))
            sections.append(((int(x1), int(y1), int(x2), int(y2)), str(ins_idx)))
            ins_idx += 1
        return (img, sections)
    detect_btn.click(run_detection, img_input, img_output)

    def select_object(evt: gr.SelectData):
        global chosen_obj
        chosen_obj = rec_list[evt.index]
        return rec_list[evt.index]

    img_output.select(select_object, None, selected_object)
    
    def run_gen(img, prompt):
        global chosen_obj
        print(prompt)
        init_img = Image.fromarray(img)
        iwidth, iheight = init_img.size
        padding_bboxes_with_color = torch.zeros(14, 20, 7)
        video_masks = torch.zeros(14, 20)
        video_masks[:, 0] = 1.0
        
        content = f'''
            Task: Automatic Trajectory Generation
            1. The user will input the current target's bbox coordinates and the resolution of the entire image.
            2. The user will provide a language description of the expected target motion.
            3. You need to directly return a list of bboxes. Please note that there should be no additional output.
            4. Return a total of 14 bboxes.
            5. Do not return anything other than the list. Strictly follow the content of the output example.
            6. Your answer must start with '[' and end with ']'.
            7. Avoid simply translating the bbox; generate diverse bbox trajectories.

            User input:
            ```
            "bbox": {chosen_obj},
            "width" : {iwidth},
            "height": {iheight},
            "text": {prompt}
            ```

            Output example:
            [
            [144, 135, 343, 222],
            [140, 135, 339, 222]
            ]
        '''

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        text = completion.choices[0].message.content
        print(text)

        box_array = normalize_coordinates(text_to_array(text), iwidth, iheight)
        padding_bboxes_with_color[:, 0, :4] = torch.tensor(box_array)
        padding_bboxes_with_color[:, :, 4] = 1.0
        
        images = pipe(
            image=init_img.resize((512, 320)),
            image_prompt=init_img,
            width=512,
            height=320,
            fps=8,
            video_masks=video_masks,
            bbox_prompt=padding_bboxes_with_color,
            num_frames=14,
            num_inference_steps=30,
            noise_aug_strength=0.02,
            motion_bucket_id=127,
            output_type='pil',
        ).frames[0]
        
        width, height = 512, 320
        frames = []
        color_lists = ["red", "yellow", "blue", "green"]
        for fid in range(14):
            img = images[fid]
            # draw = ImageDraw.Draw(img)
            # _cnt = 0
            # for bbox in padding_bboxes_with_color[fid].numpy()[:, :4]:
            #     x1, y1, x2, y2 = bbox
            #     top_left = (x1 * width, y1 * height)
            #     bottom_right = (x2 * width, y2 * height)
            #     draw.rectangle([top_left, bottom_right], outline=color_lists[_cnt % 4], width=2)
            #     _cnt += 1

            frames.append(np.array(img))
        
        export_to_video(np.array(frames), "generated.mp4")
        return gr.Video("./generated.mp4")
        
    gen_btn.click(run_gen, [img_input, text_prompt], video_output)

if __name__ == "__main__":
    demo.launch(share=True)
