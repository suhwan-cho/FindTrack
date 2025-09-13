from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModelForCausalLM
from .segment_anything import build_sam_vit_h
from .unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .configuration_evf import EvfConfig

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



class EvfSamModel(PreTrainedModel):
    config_class = EvfConfig
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(EvfSamModel, self).__init__(config)

        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.encoder_pretrained = kwargs.get("encoder_pretrained", None)
        self.initialize_evf_modules(config)

    def initialize_evf_modules(self, config):
        # SAM
        if config.sam_scale == "huge":
            self.visual_model = build_sam_vit_h(self.vision_pretrained)
        else:
            raise NotImplementedError

        # freeze the SAM encoder
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # learn others
        self.visual_model.mask_decoder.train()
        for param in self.visual_model.mask_decoder.parameters():
            param.requires_grad = True
        self.visual_model.prompt_encoder.no_mask_embed.requires_grad_(True)
            
        # beit-3
        if self.config.mm_extractor_scale == "base":
            beit_config = _get_base_config()
        elif self.config.mm_extractor_scale == "large":
            beit_config = _get_large_config()
        else:
            raise AttributeError(f"evfsam config should contain key 'mm_extractor_scale', with value 'base' or 'large'.")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if self.encoder_pretrained is not None:
            beit_state_dict = torch.load(self.encoder_pretrained)["evfsam"]
            self.mm_extractor.load_state_dict(
                beit_state_dict, 
                strict=False
            )

        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

    def inference(
            self,
            images,
            images_evf,
            input_ids,
            resize_list,
            original_size_list,
            multimask_output=False,
        ):

        with torch.no_grad():
            image_embeddings = self.visual_model.image_encoder(images)

            multimask_output = multimask_output

            output = self.mm_extractor.beit3(visual_tokens=images_evf, textual_tokens=input_ids, text_padding_position=torch.zeros_like(input_ids))

        feat = output["encoder_out"][:, :1, ...]
        feat = self.text_hidden_fcs[0](feat)
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=feat,
        )

        sparse_embeddings = sparse_embeddings.to(feat.dtype)
        low_res_masks, iou_predictions = self.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        if multimask_output:
            sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
            low_res_masks = torch.take_along_dim(low_res_masks, sorted_ids[..., None, None], dim=1)[:, :1]

        pred_mask = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=resize_list[0],
            original_size=original_size_list[0],
        )

        return pred_mask[:, 0], iou_predictions[0]


AutoConfig.register("evf", EvfConfig)
AutoModelForCausalLM.register(EvfConfig, EvfSamModel)
