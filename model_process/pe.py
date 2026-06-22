import os
import tempfile

import torch
from ultralytics import YOLO
from thop import profile


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 60)
    print(f"Total Params      : {total:,}")
    print(f"Trainable Params  : {trainable:,}")
    print(f"Total Params(M)   : {total / 1e6:.3f} M")

    return total


def get_weight_size(model):
    """
    实际权重大小
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        delete=False
    ) as f:

        torch.save(model.state_dict(), f.name)

        size_mb = os.path.getsize(
            f.name
        ) / 1024 / 1024

    os.remove(f.name)

    print("=" * 60)
    print(f"Weight Size       : {size_mb:.2f} MB")

    return size_mb


def get_model_gpu_memory(model, device):
    """
    模型加载后的显存
    """

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    before = torch.cuda.memory_allocated(device)

    model.to(device)

    torch.cuda.synchronize()

    after = torch.cuda.memory_allocated(device)

    model_mem = after - before

    print("=" * 60)
    print(
        f"Model GPU Memory  : "
        f"{model_mem / 1024**2:.2f} MB"
    )

    return model_mem


def profile_peak_memory(
    model,
    dummy_input,
    device
):
    """
    推理峰值显存
    """

    model.eval()

    dummy_input = dummy_input.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model.predict(dummy_input,        imgsz=1024,        conf=0.5,
        verbose=False)

    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated(
        device
    )

    print("=" * 60)
    print(
        f"Peak Memory       : "
        f"{peak_mem / 1024**2:.2f} MB"
    )

    return peak_mem


def count_flops(model, dummy_input):
    """
    FLOPs统计
    """

    model.eval()

    try:
        macs, params = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )

        flops = macs * 2

        print("=" * 60)
        print(
            f"FLOPs             : "
            f"{flops / 1e9:.3f} GFLOPs"
        )

        return flops

    except Exception as e:

        print("=" * 60)
        print(f"FLOPs Failed: {e}")

        return None


def profile_yolo(
    weight_path,
    imgsz=640,
    device="cuda:0"
):
    """
    YOLO模型完整统计
    """

    print("\nLoading model ...")

    yolo = YOLO(weight_path)

    # 真正的nn.Module
    model = yolo.model

    count_parameters(model)

    get_weight_size(model)

    if torch.cuda.is_available():

        get_model_gpu_memory(
            model,
            device
        )

    dummy_input = torch.randn(
        1,
        3,
        imgsz,
        imgsz
    )

    count_flops(
        model.cpu(),
        dummy_input
    )

    if torch.cuda.is_available():

        profile_peak_memory(
            model.to(device),
            dummy_input,
            device
        )

    print("=" * 60)
    print("Done")


if __name__ == "__main__":

    profile_yolo(
        weight_path=r"E:\best.pt",
        imgsz=1024,
        device="cuda:0"
    )
    """
    Loading model ...
============================================================
Total Params      : 58,789,861
Trainable Params  : 0
Total Params(M)   : 58.790 M
============================================================
Weight Size       : 224.97 MB
============================================================
Model GPU Memory  : 230.07 MB
============================================================
FLOPs             : 521.703 GFLOPs
============================================================
Peak Memory       : 674.47 MB
============================================================
Done

    """