
from typing import List, Tuple

import torch
import gradio as gr


#import utils
#myadditions
import json



#log = utils.get_pylogger(__name__)

def demo():
    
    #assert cfg.ckpt_path

    print("Running Demo")

    print("Instantiating scripted model")
    model = torch.jit.load("model.script.pt")
    model.eval()

    print(f"Loaded Model: {model}")
    with open("cifar10_classes.json") as f:
        categories = json.load(f)

    def recognize_image(image):
        if image is None:
            return None
        image = torch.tensor(image[None, ...],dtype=torch.float32)
        image = image.permute(0,3, 1, 2)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        return {categories['names'][i]: preds[i] for i in range(10)}

    im = gr.Image(shape=(32, 32), image_mode="RGB")

    demo = gr.Interface(
        fn=recognize_image,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(share=True)


def main():
    demo()

if __name__ == "__main__":
    main()