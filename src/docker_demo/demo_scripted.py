
from typing import List, Tuple

import torch
import gradio as gr
import torchvision.transforms as T


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

    def recognize_image(image):
        if image is None:
            return None
        image = T.ToTensor()(image).unsqueeze(0)
        try:
            preds = model.forward_jit(image)
        except Exception:
            traceback.print_exc()
        preds = preds[0].tolist()
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return {label[i]: preds[i] for i in range(10)}

    print(f"Loaded Model: {model}")
   

    im = gr.Image(shape=(32, 32))

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