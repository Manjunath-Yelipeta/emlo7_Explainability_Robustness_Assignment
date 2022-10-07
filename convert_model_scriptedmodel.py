import timm
import torch

def convert_model_scripted():
    model = timm.create_model("resnet18", pretrained=True)
    checkpoint_file = "./logs/train/runs/2022-10-06_05-27-40/checkpoints/epoch_009.ckpt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"],strict = False)
    model.eval()
    example = torch.rand(1, 3, 32, 32)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    scripted_module = torch.jit.script(model, example)
    #scripted_model = model.to_torchscript(method="script")
    torch.jit.save(scripted_module, "./logs/train/runs/2022-10-06_05-27-40/checkpoints/model.script.pt")

if __name__ == "__main__":
    convert_model_scripted()
    print("converted model to torch scripted model")
