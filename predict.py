import json
from typing import Any

import hydra
import numpy as np
import timm
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from timm.data.transforms_factory import transforms_imagenet_eval


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient."""
        # self.model = timm.create_model('efficientnet_b3a', pretrained=True)
        self.model = timm.create_model("resnet18", pretrained=True)
        self.checkpoint_file = "./logs/train/runs/2022-09-11_15-41-20/checkpoints/epoch_002.ckpt"
        self.checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(self.checkpoint["model"])
        self.model.eval()
        self.transform = transforms_imagenet_eval()
        with open("cifar10_classes.json") as f:
            self.labels = list(json.load(f).values())

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model."""
        # Preprocess the image
        img = Image.open(image).convert("RGB")
        img = self.transform(img)

        # Run the prediction
        with torch.no_grad():
            labels = self.model(img[None, ...])
            labels = labels[0]  # we'll only do this for one image

        # top 5 preds
        topk = labels.topk(5)[1]
        output = {
            # "labels": labels.cpu().numpy(),
            "topk": [self.labels[x] for x in topk.cpu().numpy().tolist()],
        }

        return output
