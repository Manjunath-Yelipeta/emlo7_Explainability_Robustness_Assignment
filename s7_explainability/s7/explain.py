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


from PIL import Image
from captum.robust import PGD
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
std=[0.229, 0.224, 0.225]
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            ),
        ]
    )

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_transform1 = T.Compose(
    [
        T.Lambda(lambda x: x.permute(0, 3, 1, 2)),
        T.Normalize(
            mean=(-1 * np.array(mean) / np.array(std)).tolist(),
            std=(1 / np.array(std)).tolist(),
        ),
        T.Lambda(lambda x: x.permute(0, 2, 3, 1)),
    ]
)
def image_save(img, pred,image_name):
    npimg = inv_transform(img).squeeze().permute(1, 2, 0).detach().numpy()
    plt.imshow(npimg)
    plt.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/pgd_output/' + image_name)

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

# Download human-readable labels for ImageNet.
# get the classnames
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "imagenet_classes.txt",
)
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

image_list = os.listdir("/home/ubuntu/emlo2-session4_Demo_Deployments/s7/ImageNet_10_Images/")

print("image_list: ", image_list)

for image in image_list:

    print("Processing image: ", image)

    transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])


    transform_normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    img = Image.open('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/ImageNet_10_Images/' + image)

    transformed_img = transform(img)

    img_tensor = transform_normalize(transformed_img)
    img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(device)
    output = model(img_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = categories[pred_label_idx.item()]
    # print("Predicted:", predicted_label, "(", prediction_score.squeeze().item(), ")")

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_tensor, target=pred_label_idx, n_steps=200)

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    sth = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
        use_pyplot=False,
    )

    sth[0].savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_ig.jpg")

    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        img_tensor, nt_samples=10, nt_type="smoothgrad_sq", target=pred_label_idx
    )
    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    sth[0].savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_ig_with_noisetunnel.jpg")

    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        img_tensor,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )
    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        img_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    sth = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False,
    )

    sth[0].savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_occlusion.jpg")

    # ## SHAP

    # Works well where number of classes are less
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    model_out = model(img_tensor)
    classes = torch.argmax(model_out, axis=1).cpu().numpy()
    # print(f"Classes: {classes}: {np.array(categories)[classes]}")

    img_tensor.shape

    

    def predict(imgs: torch.Tensor) -> torch.Tensor:
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(0, 3, 1, 2)

        img_tensor = imgs.to(device)

        output = model(img_tensor)
        return output

    topk = 4
    batch_size = 50
    n_evals = 10000

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))

    # create an explainer with model and image masker
    explainer = shap.Explainer(predict, masker_blur, output_names=categories)

    # feed only one image
    # here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
    # image_np = Image.open("cat.jpeg")
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    shap_values = explainer(
        img_tensor,
        max_evals=n_evals,
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk],
    )

    (shap_values.data.shape, shap_values.values.shape)

    shap_values.data = inv_transform1(shap_values.data).cpu().numpy()[0]

    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

    shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        true_labels=[categories[285]],
        show=False
    )
    plt.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + '_shap.jpg')

    # # ## Captum Robustness

    # transform = T.Compose(
    #     [
    #         T.Resize((384, 384)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)

    # img_tensor.requires_grad

    saliency = Saliency(model)
    grads = saliency.attribute(img_tensor, target=285)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    original_image = np.transpose(
        (img_tensor.squeeze(0).cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)
    )

    sth = viz.visualize_image_attr(
        None,
        original_image,
        method="original_image",
        title="Original Image",
        use_pyplot=False,
    )

    sth = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlayed Gradient Magnitudes",
        use_pyplot=False,
    )

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=285, **kwargs)

        return tensor_attributions

    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(
        ig, img_tensor, baselines=img_tensor * 0, return_convergence_delta=True
    )
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    # print("Approximation delta: ", abs(delta))

    sth = viz.visualize_image_attr(
        attr_ig,
        original_image,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title="Overlayed Integrated Gradients",
        use_pyplot=False,
    )

    sth[0].savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_saliency.jpg")
    
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    #img_tensor.requires_grad = True
    img_tensor = img_tensor.to(device)
    target_layers = [model.layer4[-1]]


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)


    targets = [ClassifierOutputTarget(int(image.split('.')[0].split('_')[-1]))]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)


    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    Image.fromarray(visualization).save('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_gradcam.jpg")
    #plt.imshow(visualization)


    from pytorch_grad_cam import GradCAMPlusPlus


    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)


    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)


    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    rgb_img = inv_transform(img_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization).save('/home/ubuntu/emlo2-session4_Demo_Deployments/s7/output_explainability/' + image.split('.')[0] + "_gradcam_plus_plus.jpg")

    
    #Use PGD to make the model predict cat for all images
    
    pgd = PGD(model, torch.nn.CrossEntropyLoss(reduction='none'), lower_bound=-1, upper_bound=1)  # construct the PGD attacker

    perturbed_image_pgd = pgd.perturb(inputs=img_tensor, radius=0.13, step_size=0.02, 
                                  step_num=7, target=torch.tensor([199]).to(device), targeted=True) 
    new_pred_pgd, score_pgd = get_prediction(model, perturbed_image_pgd)
    image_save(perturbed_image_pgd.cpu(), new_pred_pgd + " " + str(score_pgd),image.split('.')[0] + "_pgd.jpg")
    #perturbed_image_pgd.savefig('/home/ubuntu/emlo2-session4_Demo_Deployments/Explainability/s7/output_explainability/' + image + "_saliency.png")