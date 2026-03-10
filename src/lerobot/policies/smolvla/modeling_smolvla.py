#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.utils import get_safe_dtype


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    -yj
    SmolVLA에서 실제로 action chunk를 생성하는 핵심 모듈이다
    
    개인적으로 이 클래스는 크게 두 부분으로 보면 되는데
    1) image / language / state 를 prefix context로 만드는 부분
    2) noisy action을 suffix로 넣고 flow matching으로 action을 복원하는 부분 이다
    
    즉 멀티모달 정보를 조건으로 해서 action trajectory를 생성하는 구조다.
    """

    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        """
        nn.Module 초기화하는 부분
        config를 저장해두는 건 이후 hidden size, chunk size, cache 사용 여부 같은 옵션을 계속 참조해야 하기 때문이다
        여기서부터 이 클래스는 사실상 config-driven model이라고 보면 된다
        
        """

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )

        """
        이 부분이 backbone 생성하는 부분인데
        단순히 vision-language model만 쓰는 게 아니라 이름 그대로 VLM + expert 구조를 가져온다
        train_expert_only, freeze_vision_encoder 같은 옵션을 보면 이 모델은 처음부터 “전체를 다 학습할 수도 있고 일부만 학습할 수도 있게” 설계되어있다
        즉, pretrained VLM을 활용하면서 action generation 관련 부분만 학습시키려는 의도가 보인다
        """
        
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        """
        robot state는 원래 text/image token hidden size와 다르니까 projection이 필요하다
        결국 state도 prefix sequence 안에 token처럼 들어가야 하므로 hidden dimension을 맞춰주는 역할이다
        내 기준으로 이 줄은 “state를 transformer world에 넣기 위한 adapter”라고 보면 편했다.
        
        """



        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)

        """
        noisy action을 expert hidden size로 바꾸는 projection이다
        action도 결국 suffix token처럼 transformer/expert에 들어가야 해서 dim alignment가 필요하다
        raw action vector → expert token embedding이라고 이해하면 된다
        """

        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)
        """
        반대로 expert output을 다시 action dimension으로 돌려놓는 projection이다
        여기서 나오는 값은 최종 clean action 자체라기보다 학습 시에는 velocity field v_t prediction 역할을 한다
        """

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        """
        action embedding이랑 timestep embedding을 concat할 거라서 입력 차원이 2 * hidden이다
        둘을 단순 더하기가 아니라 concat 후 MLP로 섞는 구조라서 time 정보를 좀 더 명시적으로 fusion하려는 느낌
        """


        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        """
        위 MLP의 두 번째 linear layer
        action/time fusion representation을 expert input dimension에 맞게 정리해주는 역할
        diffusion/flow 모델에서 timestep conditioning이 중요하다는 점을 반영한 부분
        """


        self.set_requires_grad()
        """
        특정 모듈의 gradient on/off를 config에 맞게 적용.

        지금 코드상으로는 state_proj 쪽을 주로 제어한다.

        작은 adapter만 선택적으로 학습하려는 실험에도 대응 가능해 보인다.
        """


        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        """
        image special token id를 tokenizer에서 가져온다.

        image를 그냥 dense embedding으로만 넣는 게 아니라, language sequence와 섞일 수 있게 special token 체계를 활용한다는 뜻.
        """

        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        """
        이미지 시작을 나타내는 토큰 시퀀스.

        이미지 embedding 앞에 붙여서 “여기부터 이미지 관련 정보”라는 boundary를 만들어주는 역할.
        """

        self.add_image_special_tokens = self.config.add_image_special_tokens
        """
        이 special token을 실제로 넣을지 말지를 config로 결정.

        실험에 따라 성능 비교하려고 옵션화해둔 것 같음.
        """


        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        """
        image 끝 boundary 역할.

        시작과 끝을 둘 다 명시해서 multimodal token stream을 더 구조적으로 만들려는 의도.
        """


        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor
        """
        prefix_length: prefix sequence를 특정 길이까지 pad할 때 사용.

        rtc_processor: real-time constraint 관련 보정용 processor.

        inference 단계에서 지연이나 leftover chunk를 고려하는 분기가 뒤에 나옴.
        """

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)
        """
        PyTorch compile로 성능 최적화.

        sample_actions랑 forward 둘 다 compile하는 걸 보면, 학습과 추론 둘 다 속도 이득을 보려는 구조.

        특히 action generation은 iterative loop가 있어서 compile 이점이 있을 수 있다.
        """


    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled
    
    """
    RTC(real-time constraint) 기능이 실제로 켜져 있는지 확인

    뒤에서 inference denoising step을 일반 방식으로 할지 RTC 방식으로 할지 분기할 때 사용
    """

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
    
    """
    state_proj만 별도로 학습할지 말지 정한다

    즉 state encoder adapter를 튜닝할지 freeze할지 결정하는 줄

    개인적으로는 “전체 backbone은 건드리지 않더라도 state 쪽만 조정해볼 수 있게 해둔 장치”로 보였다
    
    """

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise
    
    """
    표준 Gaussian noise 샘플링(flow matching / diffusion 계열에서 시작점이 되는 noise 생성)

    inference에서는 이 noise에서 출발해서 action으로 가고, training에서는 clean action과 섞어서 x_t를 만든다
    """

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        """
        uniform이 아니라 Beta 분포에서 timestep 샘플링

        Beta(1.5, 1.0)이면 완전 균등은 아니고 약간 한쪽으로 치우친 샘플링이 된다

        학습에서 특정 time 구간을 조금 더 자주 보게 하려는 의도가 있을 수 있다
        """
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        """
        정확히 0이나 1이 안 되도록 살짝 shift.

        t=0, t=1은 너무 극단이라 수치적으로 애매하거나 학습이 너무 쉬워질 수 있어서 피하는 느낌

        즉 time range를 거의 [0,1]이지만 완전 끝점은 제외하는 방식
        """

        return time
    """
    최종적으로 continuous time 반환
    """

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """

        """
        이 함수는 prefix sequence를 만드는 핵심.

        prefix = image + language + state

        나중에 action suffix가 이 prefix를 condition으로 보게 된다.
        """
        embs = []
        pad_masks = []
        att_masks = []
        """
        각각

        embs: 실제 embedding token들

        pad_masks: padding 여부

        att_masks: attention block 구조를 만들기 위한 마스크용 정보

        나중에 합쳐서 전체 transformer input으로 사용.
        """


        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            """
            카메라가 여러 개일 수 있어서 image마다 반복

            strict=False는 두 iterable 길이가 완전히 같지 않아도 에러를 피하려는 선택
            """

            if self.add_image_special_tokens: 
                """
                config에서 허용한 경우에만 이미지 시작/끝 토큰을 붙인다."""
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                """
                image start token도 결국 language embedding layer를 통과시켜 hidden vector로 만든다.

                그리고 batch size만큼 expand.

                즉 special token도 일반 token embedding처럼 다뤄서 multimodal sequence에 자연스럽게 끼워 넣는다.
                """

                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                """
                special start token은 실제 존재하는 token이므로 padding mask는 전부 True.
                """

                att_masks += [0] * (image_start_mask.shape[-1])
                """
                attention block 관점에서 이 토큰들은 prefix의 앞부분에 속한다는 표시.

                뒤에서 make_att_2d_masks 할 때 cumulative 구조로 block attention이 형성된다.
                """

                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                """
                embedding과 mask를 리스트에 저장.
                """

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb
            """
            실제 이미지 embedding 생성하는 부분
            """

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            """
            embedding scale 조정하는 부분
            일반 transformer token embedding에서 hidden dim의 sqrt로 scale 맞추는 패턴과 유사하다

            image embedding magnitude를 language embedding과 어느 정도 맞춰주려는 의미로 보인다.
            """

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            """
            원래 카메라 단위 mask를 이미지 token 개수만큼 확장
            즉 이 이미지가 유효한 카메라인지 여부를 각 이미지 patch/token에 동일하게 적용
            """

            embs.append(img_emb)
            pad_masks.append(img_mask)
            """
            이미지 embedding을 prefix에 추가
            """

            att_masks += [0] * (num_img_embs)
            """
            이미지 token들도 prefix block의 일부로 추가
            """

            if self.add_image_special_tokens:
                """
                이미지 끝 토큰도 넣을지 확인
                """

                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                """
                end token도 language embedding 방식으로 hidden vector화

                이미지 시작과 동일한 처리 흐름
                """

                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                """
                end token 역시 valid token
                """

                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
                """
                이미지 끝 boundary까지 prefix에 포함
                """

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        """
        language token embedding => instruction, task description 같은 텍스트 조건이 여기에 해당함
        """

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        """
        language embedding도 scaling 적용함

        image embedding과 유사하게 hidden magnitude를 맞추는 느낌임
        """

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        """
        language token들을 prefix에 추가
        """

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        """
        language 역시 prefix block

        아직까지 image/language는 같은 앞부분 block으로 묶여 있다
        """

        state_emb = self.state_proj(state)
        """
        raw robot state를 hidden dimension으로 projection.

        이걸 통해 state도 transformer가 읽을 수 있는 token 표현으로 바꾼다."""

        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        """
        state가 (B, H)면 (B, 1, H)로 바꿔서 sequence token처럼 맞춘다 => 즉 state는 보통 1개의 token 역할
        """

        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device
        """
        state embedding도 prefix 끝에 붙인다

        이제 전체 prefix는 image → language → state 순서가 된다
        """

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        """
        state token은 실제 입력이므로 valid mask를 준다
        """

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        """
        여기서 attention block이 바뀐다(image/language는 0, state는 1)

        => 즉 state부터는 prefix 내부에서도 새 attention boundary가 생긴다.

        이 코드는 처음 보면 헷갈리는데, 뒤의 cumulative mask 생성 때문에 “같은 숫자 블록끼리 / 이전 블록을 보는 방식”으로 attention을 조절한다.
        """

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        """
        지금까지 모은 prefix token들을 하나의 sequence로 합침
        """

        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]
        """
        Python list로 관리하던 attention structure를 tensor로 변환, batch 차원을 맞추기 전에 (1, L) 형태로 만들어둔다
        """

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

            """
            prefix 길이가 고정 길이보다 짧으면 padding(shape을 일정하게 맞추기 위해)

            특히 multimodal 입력 길이가 매번 달라질 수 있으니 미리 일정 길이로 맞추는 전략.
            """
        att_masks = att_masks.expand(bsize, -1) 
        """
        batch 크기만큼 attention mask 확장
        """

        return embs, pad_masks, att_masks
    """
    prefix embedding, padding mask, attention block 정보를 반환.

    이후 suffix와 합쳐서 전체 transformer input을 구성한다.
    """


    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing"""
        """
        suffix는 action generation 대상

        현재 noisy action과 timestep을 함께 넣어 denoising step에 필요한 입력을 만든다"""

        embs = []
        pad_masks = []
        att_masks = []
        """
        prefix와 동일하게 embedding / pad / attention 구조를 따로 관리
        """

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        """
        noisy action chunk를 hidden size로 projection.
        각 time step의 action이 하나의 token처럼 바뀐다고 생각하면 된다.
        """

        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        """
        이후 timestep embedding 생성과 mask 생성에 필요한 메타정보 저장
        """

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        """
        scalar timestep을 sinusoidal embedding으로 바꾼다.

        여기서 timestep은 token position이 아니라 flow time t.

        diffusion에서 time embedding 넣는 것과 같은 개념인데, 여기선 flow matching용.
        """

        time_emb = time_emb.type(dtype=dtype)
        """
        action embedding dtype과 맞춰준다

        mixed precision 환경에서 dtype mismatch를 피하려는 처리
        """

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        """
        (B, H)인 time embedding을 (B, chunk_size, H)로 확장 => 즉 action chunk의 모든 step이 같은 현재 timestep 정보를 공유한다.
        """

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        """
        action 정보와 time 정보를 concat => 둘을 단순 더하는 게 아니라 concat 후 MLP로 섞겠다는 의도.
        """

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        """
        action/time fusion MLP.

        SiLU 활성화는 diffusion류 모델에서 자주 쓰이는 편이라 자연스럽다.

        이 과정을 거쳐 “현재 시점의 noisy action token” 표현이 만들어진다.
        """

        # Add to input tokens
        embs.append(action_time_emb)
        """
        suffix token sequence에 추가
        """

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        """
        action chunk의 각 token은 모두 실제 입력이므로 valid mask = True
        """

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        """
        suffix는 state와 같은 block id를 쓰는 구조로 보인다.

        결과적으로 prefix/suffix attention 구조를 cumulative rule로 만들 수 있게 함.

        특히 suffix 내부에서 causal/block attention을 만들기 위한 준비 단계.
        """

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        """
        suffix도 하나의 sequence로 합침
        """


        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        """
        주의할 점은 여기 dtype이 bool이 아니라 embedding dtype을 따라간다.

        make_att_2d_masks에서 cumsum을 쓰기 때문에 수치형이어도 동작 가능하다.

        prefix 쪽과 구현 스타일이 조금 다른데 기능상 큰 문제는 없어 보인다.
        """

        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        """
        batch 차원으로 확장
        """
        return embs, pad_masks, att_masks
    """
    suffix embedding / mask 반환.
    """
    


    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        """
        이 forward는 training용

        clean action이 주어졌을 때 noise와 섞어서 flow matching loss를 계산한다
        """

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        """
        외부에서 noise를 주지 않으면 내부에서 랜덤 샘플링.

        shape은 action과 동일해야 elementwise interpolation 가능.
        """

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        """
        batch마다 하나의 continuous time t를 뽑음 => shape은 (B,)
        """

        time_expanded = time[:, None, None]
        """
        action tensor와 브로드캐스팅하려고 (B,1,1)로 바꾼다
        """

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        """
        flow matching에서 사용하는 interpolation point
        t=0이면 clean action, t=1이면 pure noise
        중간 시점의 sample x_t를 만든다
        """

        u_t = noise - actions
        """
        target velocity(이 모델이 예측해야 하는 벡터장 target이다)
        여기서 flow matching이 diffusion이랑 다른 느낌을 주는 핵심이 드러난다.
        """

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        """
        조건 정보(prefix) 생성 => image, language, state가 여기에 들어감
        """

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        """
        noisy action과 timestep을 suffix로 임베딩
        """

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        """
        prefix와 suffix를 하나의 긴 sequence로 결합, pad mask와 attention block 정보도 같이 결합
        """

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        """
        1D block attention 정보를 실제 2D attention mask로 바꾸는 단계
        transformer에 넣을 수 있는 형태의 attention mask 생성
        """

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        """
        valid token 기준으로 position id를 만든다.
        padding은 position 증가에 영향을 덜 주게 처리.
        """


        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        """
        학습에서는 prefix와 suffix를 같이 넣고 한 번에 forward.
        cache는 사용하지 않는다. 학습에서는 iterative inference처럼 prefix를 재사용할 필요가 없기 때문.
        출력 중 suffix 쪽 hidden만 나중에 action prediction에 사용.
        """


        suffix_out = suffix_out[:, -self.config.chunk_size :]
        """
        suffix 중에서도 실제 action chunk 길이만큼만 사용.
        혹시 내부 처리상 더 긴 부분이 있어도 마지막 chunk_size만 남긴다.
        """

        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        """
        최종 regression 전에 float32로 upcast.
        mixed precision 상태에서 MSE 계산 안정성을 높이려는 의도.
        """

        v_t = self.action_out_proj(suffix_out)
        """
        expert hidden을 action dim으로 projection해서 velocity prediction 생성.
        """

        losses = F.mse_loss(u_t, v_t, reduction="none")
        """
        expert hidden을 action dim으로 projection해서 velocity prediction 생성.
        """

        return losses
    """
    (B, chunk_size, action_dim) 형태 loss 반환.

    실제 평균은 바깥 policy wrapper에서 처리.
    """

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        """
        inference용 action generation 함수
        training과 달리 clean action이 없으므로 pure noise에서 시작해서 iterative denoising을 수행한다
        """

        bsize = state.shape[0]
        device = state.device
        """
        배치 크기와 device 저장
        """

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        """
        시작점은 random noise
        출력 action chunk와 같은 shape로 초기화.
        """

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        """
        조건 context(prefix)를 만든다
        inference에서도 이 부분은 동일
        """

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        """
        prefix만으로 attention mask 생성
        """

        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        """
        prefix용 position id 생성
        """


        # Compute image and language key value cache
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        """
        여기서 inference 최적화 포인트가 나온다.

        prefix는 denoising step마다 안 바뀌니까 한 번만 forward해서 KV cache를 채운다.

        이후 반복문에서는 suffix만 새로 넣고 prefix는 cache 재사용.

        이게 없으면 denoising step마다 이미지/언어/state를 전부 다시 처리해야 해서 비효율적이다.
        """

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        """
        Euler-like integration step size.\

        음수인 이유는 time이 1 → 0 방향으로 내려가야 하기 때문

        즉 noise 쪽에서 action 쪽으로 역방향 integration
        """

        x_t = noise 
        """
        초기 상태는 pure noise
        """

        for step in range(num_steps):
            time = 1.0 + step * dt
            """
            denoising loop.

            step=0일 때 time≈1, 마지막에는 time≈0에 가까워짐.

            flow trajectory를 따라 내려가는 구조.
            """


            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            """
            현재 scalar timestep을 batch 형태 (B,)로 확장.
            suffix embedding 만들 때 필요.
            """

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )
            """
            현재 timestep과 prefix cache를 고정한 채 denoise_step을 부르는 partial wrapper 느낌.
            RTC processor가 이 함수를 감싸서 사용할 수 있게 만든 구조라 이해하면 된다.
            """

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                """
                RTC 모드일 때 추가 정보 가져오기.
                실시간 제어에서는 이전 chunk leftover나 실행 지연이 실제 action quality에 영향 줄 수 있어서 이런 인자가 필요해 보인다.
                """

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
                """
                RTC 모드에서는 일반 denoise 대신 processor가 감싼 버전을 사용.
                즉 모델 자체는 같지만 실제 제어 시점 제약을 반영하는 후처리/보정 레이어가 들어가는 구조.
                """

            else:
                v_t = denoise_step_partial_call(x_t)
            """
            RTC가 꺼져 있으면 일반적인 velocity prediction 사용
            """

            x_t = x_t + dt * v_t
            """
            Euler update
            현재 상태에서 velocity field를 따라 조금 이동
            dt가 음수이므로 결과적으로 noise → action 방향으로 이동하게 된다
            """

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
            """
            디버깅용 tracking
            각 timestep에서 상태가 어떻게 변하는지 저장해서 나중에 분석할 수 있게 함.
            """

        return x_t
    """
    모든 denoising step이 끝난 최종 action chunk 반환.

    이 시점의 x_t는 더 이상 noise가 아니라 action sample로 해석된다.
    """

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        """
        inference 중 한 스텝만 수행하는 함수이다. 현재 noisy action x_t를 넣으면 현재 timestep의 velocity v_t를 예측한다
        """

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
