## Importing Libraries
import os
import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)

import shap
import timm
import torch
import urllib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from captum.attr import FeatureAblation
from captum.robust import MinParamPerturbation
import albumentations as A
import albumentations.pytorch.transforms as AT


from PIL import Image
from captum.robust import PGD, FGSM
from matplotlib.colors import LinearSegmentedColormap
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from captum.attr import (
    DeepLift,
    Saliency,
    Occlusion,
    NoiseTunnel,
    GradientShap,
    IntegratedGradients,
    visualization as viz,
)

device = torch.device("cpu")
model = timm.create_model("resnet18", pretrained=True)
model.eval()
model = model.to(device)
mean=[0.485, 0.456, 0.406]


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def get_prediction(model, image: torch.Tensor):
    model = model.to(device)
    img_tensor = image.to(device)
    with torch.no_grad():
        output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]

    return predicted_label, prediction_score.squeeze().item()


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
inv_transform= T.Compose([
T.Normalize(
    mean = (-1 * np.array(mean) / np.array(std)).tolist(),
    std = (1 / np.array(std)).tolist()
),
])
# Download human-readable labels for ImageNet.
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]





def pixel_dropout(image, dropout_pixels):
    keep_pixels = image[0][0].numel() - int(dropout_pixels)
    vals, _ = torch.kthvalue(pixel_attr.flatten(), keep_pixels)
    return (pixel_attr < vals.item()) * image

class AlbumentationTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)

        return self.transforms(image=img)['image']



for image in os.listdir('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/ImageNet_10_Images/'):
    # if i == 'i4.png':
    #     continue
    img = Image.open('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/ImageNet_10_Images/' + image)
    # print dimension of img
    print("image name: ", image, " and size: ", img.size)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    # Get original prediction
    pred, score  = get_prediction(model, img_tensor)

    # Construct FGSM attacker
    fgsm = FGSM(model, lower_bound=-1, upper_bound=1)
    perturbed_image_fgsm = fgsm.perturb(img_tensor, epsilon=0.16, target=285) 
    new_pred_fgsm, score_fgsm = get_prediction(model, perturbed_image_fgsm)

    #image_show(perturbed_image_fgsm.cpu(), new_pred_fgsm + " " + str(score_fgsm))

    fig1 = plt.gcf()
    npimg = inv_transform(perturbed_image_fgsm.cpu()).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % new_pred_fgsm + " " + str(score_fgsm))
    plt.show()
    fig1.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/Robustness_output/' + image.split('.')[0] + "_fgsm.jpg", dpi=100)
    #Feature Ablation
    feature_mask = torch.arange(64 *7*7).reshape(8*7, 8*7).repeat_interleave(repeats=4, dim=1).repeat_interleave(repeats=4, dim=0).reshape(1, 1, 224,224)
    print(feature_mask)

    feature_mask.shape

    type(feature_mask)

    

    ablator = FeatureAblation(model.cpu())
    attr = ablator.attribute(img_tensor.cpu(), target=285, feature_mask=feature_mask)
    # Choose single channel, all channels have same attribution scores
    pixel_attr = attr[:,0:1]

    

    

    min_pert_attr = MinParamPerturbation(forward_func=model, attack=pixel_dropout, arg_name="dropout_pixels", mode="linear",
                                        arg_min=0, arg_max=1024, arg_step=16,
                                        preproc_fn=None, apply_before_preproc=True)

    pixel_dropout_im, pixels_dropped = min_pert_attr.evaluate(img_tensor.cpu(), target=285, perturbations_per_eval=10)
    print("Minimum Pixels Dropped:", pixels_dropped)

    new_pred_dropout, score_dropout = get_prediction(model, pixel_dropout_im)
    #image_show(pixel_dropout_im.cpu(), new_pred_dropout + " " + str(score_dropout))
    fig1 = plt.gcf()
    npimg = inv_transform(pixel_dropout_im.cpu()).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % new_pred_dropout + " " + str(score_dropout))
    plt.show()
    fig1.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/Robustness_output/' + image.split('.')[0]  + "_pixeldropout_.jpg", dpi=100)
    #adding random noise 
    transform1 =A.Compose([ A.Resize(224, 224),
            A.GaussNoise(p=0.8),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            AT.ToTensorV2()
        ])
    
    
#img = Image.open('cat.jpeg')

    gauss_noise = AlbumentationTransforms(transform1)
    img_tensor = gauss_noise(img)
    img_tensor = img_tensor.unsqueeze(0)
    #img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)
    # Get original prediction
    pred, score  = get_prediction(model, img_tensor)

    #image_show(img_tensor.cpu(), pred + " " + str(score))
    fig1 = plt.gcf()
    npimg = inv_transform(img_tensor.cpu()).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred + " " + str(score))
    plt.show()
    fig1.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/Robustness_output/' + image.split('.')[0] + "_Gaussnoise.jpg", dpi=100)

    #adding RandomBrightness
    transform2 =A.Compose([ A.Resize(224, 224),
            A.RandomBrightnessContrast(p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            AT.ToTensorV2()
        ])
    #img = Image.open('cat.jpeg')

    random_brightness = AlbumentationTransforms(transform2)
    img_tensor = random_brightness(img)
    img_tensor = img_tensor.unsqueeze(0)
    #img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)
    # Get original prediction
    pred, score  = get_prediction(model, img_tensor)

    #image_show(img_tensor.cpu(), pred + " " + str(score))
    fig1 = plt.gcf()
    npimg = inv_transform(img_tensor.cpu()).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.title("prediction: %s" % pred + " " + str(score))
    plt.show()
    fig1.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/Robustness_output/' + image.split('.')[0]  + "_RandomBrightness.jpg", dpi=100)




