import os
import cv2
import numpy as np
import torch 
from einops import rearrange
from torch.nn import functional as F

def show_mask(mask_image, cls_id):
    """
    QL modify
    在Matplotlib的坐标轴上叠加显示掩膜
    :param mask: 二值掩膜（HxW格式，0或1的布尔/整型数组）
    :param ax: Matplotlib的坐标轴对象
    :param random_color: 是否随机生成颜色（默认使用固定颜色）
    """
    mask = mask_image.squeeze().cpu().numpy()
    mask = mask.copy() # not writable np warning
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    filename = f"binary_mask_{cls_id.cpu().numpy().astype(np.uint8)}.png"
    cv2.imwrite(os.path.join(r"D:\301-Task2\binary_mask_model_forward", filename), binary_mask)
    
# forward process of the model
def model_forward_function(prototype_prompt_encoder, 
                            sam_prompt_encoder, 
                            sam_decoder, 
                            sam_feats, 
                            prototypes, 
                            cls_ids): 
        
    sam_feats = rearrange(sam_feats, 'b h w c -> b (h w) c')

    
    dense_embeddings, sparse_embeddings = prototype_prompt_encoder(sam_feats, prototypes, cls_ids)

    pred = []
    pred_quality = []
    sam_feats = rearrange(sam_feats,'b (h w) c -> b c h w', h=64, w=64)
 
    for dense_embedding, sparse_embedding, features_per_image in zip(dense_embeddings.unsqueeze(1), sparse_embeddings.unsqueeze(1), sam_feats):    
        
        low_res_masks_per_image, mask_quality_per_image = sam_decoder(
                image_embeddings=features_per_image.unsqueeze(0),
                image_pe=sam_prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embedding,
                dense_prompt_embeddings=dense_embedding, 
                multimask_output=False,
            )
        
        # QL modify "可视化low_res_masks_per_image"
        # save_dir = r"./low_res_masks_vis"
        # mask_vis = low_res_masks_per_image.detach().cpu().numpy()
        # mask_vis = np.squeeze(mask_vis)  # 去除多余维度，通常变成(H, W)        
        # 归一化到0~1
        # mask_norm = (mask_vis - mask_vis.min()) / (mask_vis.max() - mask_vis.min() + 1e-8)
        # 显示
        # plt.imshow(mask_norm, cmap='gray')
        # plt.title("low_res_masks_per_image")
        # plt.axis('off')
        # plt.show()
        # mask_img = (mask_norm * 255).astype(np.uint8)
        # print(f"int(cls_ids.item()): {int(cls_ids.item())}")
        # save_path = os.path.join(save_dir, f"low_res_masks_per_image_{int(cls_ids.item())}.png")
        # Image.fromarray(mask_img).save(save_path)

        pred_per_image = postprocess_masks(
            low_res_masks_per_image,
            # input_size (tuple(int, int)): The size of the image input to the
            # model, in (H, W) format. Used to remove padding.
            # original_size (tuple(int, int)): The original size of the image
            # before resizing for input to the model, in (H, W) format.
            # input_size是输入模型的预处理的图像尺寸，original_size是图像在输入模型之前的原始尺寸。
            input_size=(535, 1024), # QL modify input_size=(819, 1024),
            original_size=(1004, 1920) # QL modify original_size=(1024, 1280),
        )
        # show_mask(pred_per_image, cls_ids)
        pred.append(pred_per_image)
        pred_quality.append(mask_quality_per_image)
        
    pred = torch.cat(pred,dim=0).squeeze(1)

    # QL modify "保存pred为图片"
    # save_dir = r"./pred_masks"
    # # 清空输出目录
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.makedirs(save_dir, exist_ok=True)
    
    # pred_img = (pred.sigmoid().cpu().numpy() * 255).astype(np.uint8) if hasattr(pred, 'sigmoid') else (pred.numpy() * 255).astype(np.uint8)
    # pred_img = np.squeeze(pred_img)
    # pred_img_path = os.path.join(save_dir, "pred.png")
    # from PIL import Image
    # Image.fromarray(pred_img).save(pred_img_path)

    pred_quality = torch.cat(pred_quality,dim=0).squeeze(1)
    
    return pred, pred_quality



# taken from sam.postprocess_masks of https://github.com/facebookresearch/segment-anything
def postprocess_masks(masks, input_size, original_size):
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    # masks = F.interpolate(
    #     masks,
    #     (1024, 1024)
    #     mode="bilinear",
    #     align_corners=False,
    # )
    # 没有预处理图像，注释掉两行直接上采样到原始尺寸（跳过中间处理）
    masks = masks[..., : input_size[0], : input_size[1]]
    # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    masks = F.interpolate(
        masks,
        size=original_size,  # 注意：PyTorch 尺寸格式为 (height, width)
        mode="nearest",      # 必须使用 nearest 保持类别标签
        align_corners=None   # 对于 nearest 模式应设置为 None
    )
    
    return masks

