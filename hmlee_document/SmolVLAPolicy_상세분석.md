# SmolVLAPolicy 클래스 상세 분석

## 1. 헤더 및 임포트 (1-72줄)

### 임포트 섹션
```python
import math  # 수학 함수 (sqrt 등)
from collections import deque  # 큐 자료구조 (액션 버퍼링용)
from typing import TypedDict, Unpack  # 타입 힌팅
import torch  # PyTorch
import torch.nn.functional as F  # PyTorch 함수형 API
from torch import Tensor, nn  # 텐서와 신경망 모듈
```

### 주요 상수 및 유틸리티
- `ACTION`: 액션 키 상수
- `OBS_STATE`: 상태 관찰 키
- `OBS_LANGUAGE_TOKENS`: 언어 토큰 키
- `OBS_LANGUAGE_ATTENTION_MASK`: 언어 어텐션 마스크 키

---

## 2. 헬퍼 함수들 (74-222줄)

### ActionSelectKwargs (74-78줄)
```python
class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None  # 추론 지연 시간
    prev_chunk_left_over: Tensor | None  # 이전 청크의 남은 부분
    execution_horizon: int | None  # 실행 시간 범위
```
- RTC(Real-Time Chunking) 관련 옵션을 타입 안전하게 전달

### create_sinusoidal_pos_embedding (80-98줄)
```python
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
```
- **목적**: Flow Matching의 timestep을 sinusoidal positional embedding으로 변환
- **동작**:
  - `dimension`이 짝수인지 확인 (sin/cos 쌍 필요)
  - `time`이 1D 텐서인지 확인
  - `min_period`와 `max_period` 사이의 주기 범위 생성
  - 각 timestep에 대해 sin/cos embedding 계산
- **사용**: Flow Matching에서 timestep 정보를 액션 expert에 전달

### make_att_2d_masks (101-131줄)
```python
def make_att_2d_masks(pad_masks, att_masks):
```
- **목적**: 2D attention mask 생성 (어떤 토큰이 어떤 토큰을 볼 수 있는지)
- **동작**:
  - `pad_masks`: 패딩된 토큰 표시 (bool[B, N])
  - `att_masks`: attention 제약 표시 (int[B, N])
  - `cumsum`으로 누적합 계산 → causal attention 패턴 생성
  - 패딩 마스크와 결합하여 최종 2D mask 생성
- **예시**:
  - `[1 1 1 1 1 1]`: 완전한 causal attention
  - `[0 0 0 1 1 1]`: prefix-LM attention (앞부분은 bidirectional, 뒷부분은 causal)

### resize_with_pad (134-153줄)
```python
def resize_with_pad(img, width, height, pad_value=-1):
```
- **목적**: 이미지를 비율 유지하며 리사이즈하고 패딩
- **동작**:
  - 현재 이미지 크기와 목표 크기의 비율 계산
  - 더 큰 비율을 사용하여 비율 유지
  - bilinear interpolation으로 리사이즈
  - 왼쪽과 위쪽에 패딩 추가 (pad_value=-1)
- **사용**: SigLIP vision encoder 입력 준비

### pad_vector (156-167줄)
```python
def pad_vector(vector, new_dim):
```
- **목적**: 벡터를 `new_dim` 크기로 패딩
- **동작**:
  - 이미 `new_dim`이면 그대로 반환
  - 아니면 0으로 채운 새 텐서 생성 후 앞부분에 원본 복사
- **사용**: state/action을 `max_state_dim`/`max_action_dim`으로 패딩

### Aloha 그리퍼 변환 함수들 (184-221줄)
- **aloha_gripper_to_angular**: Aloha의 선형 그리퍼 값을 각도 공간으로 변환
- **aloha_gripper_from_angular**: 각도 공간에서 Aloha 형식으로 변환
- **aloha_gripper_from_angular_inv**: 역변환
- **목적**: Aloha 로봇과 SmolVLA의 그리퍼 표현 차이 해결

---

## 3. SmolVLAPolicy 클래스 (224-503줄)

### 클래스 정의 (224-229줄)
```python
class SmolVLAPolicy(PreTrainedPolicy):
    config_class = SmolVLAConfig  # 설정 클래스 지정
    name = "smolvla"  # 정책 이름
```
- `PreTrainedPolicy` 상속: pretrained 모델 로딩/저장 기능 제공

### __init__ 메서드 (231-247줄)
```python
def __init__(self, config: SmolVLAConfig, **kwargs):
    super().__init__(config)  # 부모 클래스 초기화
    config.validate_features()  # 설정 검증
    self.config = config  # 설정 저장
    self.init_rtc_processor()  # RTC 프로세서 초기화
    self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)  # 실제 모델 생성
    self.reset()  # 큐 초기화
```

**단계별 설명**:
1. `super().__init__(config)`: PreTrainedPolicy 초기화
2. `config.validate_features()`: 설정의 feature 정의 검증
3. `self.config = config`: 설정을 인스턴스 변수로 저장
4. `self.init_rtc_processor()`: Real-Time Chunking 프로세서 초기화 (선택적)
5. `self.model = VLAFlowMatching(...)`: 실제 모델 인스턴스 생성
6. `self.reset()`: 액션 큐 초기화

### reset 메서드 (249-253줄)
```python
def reset(self):
    """환경 리셋 시 호출"""
    self._queues = {
        ACTION: deque(maxlen=self.config.n_action_steps),
    }
```
- **목적**: 환경이 리셋될 때 액션 큐 초기화
- **동작**: `n_action_steps` 크기의 deque 생성 (최대 크기 제한)

### init_rtc_processor 메서드 (255-269줄)
```python
def init_rtc_processor(self):
    self.rtc_processor = None  # 기본값은 None
    
    if self.config.rtc_config is not None:  # RTC 설정이 있으면
        self.rtc_processor = RTCProcessor(self.config.rtc_config)  # 프로세서 생성
        
        # 모델이 이미 생성된 경우 (나중에 호출된 경우)
        model_value = getattr(self, "model", None)
        if model_value is not None:
            model_value.rtc_processor = self.rtc_processor  # 모델에 연결
```
- **목적**: Real-Time Chunking 프로세서 초기화
- **RTC**: 실시간 추론 시 이전 청크와의 연속성 유지

### get_optim_params 메서드 (271-272줄)
```python
def get_optim_params(self) -> dict:
    return self.parameters()  # 모든 파라미터 반환
```
- **목적**: 최적화에 사용할 파라미터 반환
- **사용**: 학습 시 optimizer에 전달

### _get_action_chunk 메서드 (274-302줄)
```python
def _get_action_chunk(
    self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
) -> Tensor:
```
- **목적**: 액션 청크 전체를 생성하는 내부 메서드

**단계별 설명**:
1. **큐에서 데이터 스택** (282-284줄):
   ```python
   for k in batch:
       if k in self._queues and k != ACTION:
           batch[k] = torch.stack(list(self._queues[k]), dim=1)
   ```
   - 큐에 있는 관찰 데이터를 배치로 변환
   - ACTION은 제외 (추론 시 없음)

2. **데이터 전처리** (286-289줄):
   ```python
   images, img_masks = self.prepare_images(batch)  # 이미지 전처리
   state = self.prepare_state(batch)  # 상태 패딩
   lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]  # 언어 토큰
   lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]  # 언어 마스크
   ```

3. **모델로 액션 생성** (291-293줄):
   ```python
   actions = self.model.sample_actions(
       images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
   )
   ```

4. **액션 언패딩** (295-297줄):
   ```python
   original_action_dim = self.config.action_feature.shape[0]
   actions = actions[:, :, :original_action_dim]
   ```
   - 패딩 제거하여 원본 액션 차원으로 복원

5. **Aloha 변환** (299-301줄):
   ```python
   if self.config.adapt_to_pi_aloha:
       actions = self._pi_aloha_encode_actions(actions)
   ```

### _prepare_batch 메서드 (304-308줄)
```python
def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    if self.config.adapt_to_pi_aloha:
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
    return batch
```
- **목적**: 배치 데이터 전처리
- **동작**: Aloha 모드일 때 상태를 SmolVLA 형식으로 변환

### predict_action_chunk 메서드 (310-320줄)
```python
@torch.no_grad()  # 그래디언트 계산 비활성화
def predict_action_chunk(
    self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
) -> Tensor:
    self.eval()  # 평가 모드로 전환
    
    batch = self._prepare_batch(batch)  # 배치 전처리
    self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])  # 큐 업데이트
    
    actions = self._get_action_chunk(batch, noise, **kwargs)  # 액션 생성
    return actions
```
- **목적**: 액션 청크 전체를 예측 (추론용)
- **특징**: 
  - `@torch.no_grad()`: 메모리 절약 및 속도 향상
  - 큐를 업데이트하여 시퀀스 정보 유지

### select_action 메서드 (322-348줄)
```python
@torch.no_grad()
def select_action(
    self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs
) -> Tensor:
```
- **목적**: 단일 액션 선택 (환경 실행용)
- **동작**:
  1. RTC 모드 체크 (333-335줄): RTC는 지원 안 함
  2. 평가 모드 전환 (337줄)
  3. 배치 전처리 및 큐 업데이트 (338-339줄)
  4. 큐가 비어있으면 새 액션 생성 (341-346줄):
     ```python
     if self._check_get_actions_condition():
         actions = self._get_action_chunk(batch, noise)
         # 큐 형식: (n_action_steps, batch_size, action_dim)
         self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
     ```
  5. 큐에서 하나 꺼내서 반환 (348줄)

**큐 관리 전략**:
- 모델은 한 번에 `n_action_steps`개의 액션 생성
- 큐에 저장 후 하나씩 반환하여 효율성 향상

### _check_get_actions_condition 메서드 (350-351줄)
```python
def _check_get_actions_condition(self) -> bool:
    return len(self._queues[ACTION]) == 0
```
- **목적**: 새 액션을 생성해야 하는지 확인
- **조건**: 큐가 비어있을 때만 True

### _rtc_enabled 메서드 (353-354줄)
```python
def _rtc_enabled(self) -> bool:
    return self.config.rtc_config is not None and self.config.rtc_config.enabled
```
- **목적**: RTC가 활성화되어 있는지 확인

### forward 메서드 (356-401줄) - 학습용
```python
def forward(
    self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
) -> dict[str, Tensor]:
```
- **목적**: 학습 시 loss 계산

**단계별 설명**:

1. **Aloha 변환** (369-371줄):
   ```python
   if self.config.adapt_to_pi_aloha:
       batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
       batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
   ```

2. **데이터 준비** (373-377줄):
   ```python
   images, img_masks = self.prepare_images(batch)
   state = self.prepare_state(batch)
   lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
   lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
   actions = self.prepare_action(batch)
   ```

3. **모델 forward 및 loss 계산** (378-381줄):
   ```python
   actions_is_pad = batch.get("actions_id_pad")
   loss_dict = {}
   losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
   loss_dict["losses_after_forward"] = losses.clone().mean().item()
   ```

4. **에피소드 경계 마스킹** (383-386줄):
   ```python
   if actions_is_pad is not None:
       in_episode_bound = ~actions_is_pad  # 에피소드 내부만 True
       losses = losses * in_episode_bound.unsqueeze(-1)  # 패딩된 부분 제외
       loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()
   ```

5. **패딩 제거** (388-390줄):
   ```python
   losses = losses[:, :, : self.config.max_action_dim]
   loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()
   ```

6. **Loss reduction** (392-401줄):
   ```python
   if reduction == "none":
       # 샘플별 loss 반환 (RA-BC weighting용)
       per_sample_loss = losses.mean(dim=(1, 2))  # (B,)
       loss_dict["loss"] = per_sample_loss.mean().item()
       return per_sample_loss, loss_dict
   else:
       # 스칼라 평균 loss 반환
       loss = losses.mean()
       loss_dict["loss"] = loss.item()
       return loss, loss_dict
   ```

### prepare_images 메서드 (403-443줄)
```python
def prepare_images(self, batch):
    """이미지 전처리: 리사이즈, 패딩, 정규화"""
```

**단계별 설명**:

1. **초기화** (407-408줄):
   ```python
   images = []
   img_masks = []
   ```

2. **배치에서 이미지 키 확인** (409-410줄):
   ```python
   present_img_keys = [key for key in self.config.image_features if key in batch]
   missing_img_keys = [key for key in self.config.image_features if key not in batch]
   ```

3. **에러 체크** (412-415줄):
   ```python
   if len(present_img_keys) == 0:
       raise ValueError(...)  # 최소 하나의 이미지는 필요
   ```

4. **존재하는 이미지 전처리** (417-432줄):
   ```python
   for key in present_img_keys:
       img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
       # 5D면 마지막 프레임만 사용, 4D면 그대로 사용
       
       if self.config.resize_imgs_with_padding is not None:
           img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
       
       # [0,1] → [-1,1] 정규화 (SigLIP 요구사항)
       img = img * 2.0 - 1.0
       
       # 패딩 마스크 처리
       if f"{key}_padding_mask" in batch:
           mask = batch[f"{key}_padding_mask"].bool()
       else:
           mask = torch.ones(bsize, dtype=torch.bool, device=device)
       
       images.append(img)
       img_masks.append(mask)
   ```

5. **누락된 카메라 처리** (434-442줄):
   ```python
   for num_empty_cameras in range(len(missing_img_keys)):
       if num_empty_cameras >= self.config.empty_cameras:
           break  # 설정된 개수만큼만 빈 이미지 생성
       img = torch.ones_like(img) * -1  # -1로 채운 빈 이미지
       mask = torch.zeros_like(mask)  # 마스크는 False
       images.append(img)
       img_masks.append(mask)
   ```

### Aloha 변환 메서드들 (445-470줄)

#### _pi_aloha_decode_state (445-452줄)
```python
def _pi_aloha_decode_state(self, state):
    # 특정 조인트 반전 (Aloha → SmolVLA)
    for motor_idx in [1, 2, 8, 9]:
        state[:, motor_idx] *= -1
    # 그리퍼: 선형 → 각도 변환
    for motor_idx in [6, 13]:
        state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
    return state
```
- **목적**: Aloha 상태를 SmolVLA 형식으로 변환

#### _pi_aloha_encode_actions (454-461줄)
```python
def _pi_aloha_encode_actions(self, actions):
    # 조인트 반전
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    # 그리퍼: 각도 → Aloha 형식
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
    return actions
```
- **목적**: SmolVLA 액션을 Aloha 형식으로 변환 (추론 시)

#### _pi_aloha_encode_actions_inv (463-470줄)
```python
def _pi_aloha_encode_actions_inv(self, actions):
    # 역변환 (학습 시 사용)
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
    return actions
```

### prepare_state 메서드 (472-476줄)
```python
def prepare_state(self, batch):
    """상태 패딩"""
    state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
    # 3D면 마지막 프레임만, 2D면 그대로
    state = pad_vector(state, self.config.max_state_dim)
    return state
```

### prepare_action 메서드 (478-481줄)
```python
def prepare_action(self, batch):
    """액션 패딩"""
    actions = pad_vector(batch[ACTION], self.config.max_action_dim)
    return actions
```

### PEFT 관련 메서드들 (483-503줄)

#### _get_default_peft_targets (483-492줄)
```python
def _get_default_peft_targets(self) -> dict[str, any]:
    """PEFT 타겟 모듈 반환"""
    common_projections = (
        "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
    )
    target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
    return {
        "target_modules": target_modules,
        "modules_to_save": [],
    }
```
- **목적**: LoRA 등 PEFT의 타겟 모듈 지정
- **타겟**: Expert의 Q/V projection + 공통 projection 레이어

#### _validate_peft_config (494-503줄)
```python
def _validate_peft_config(self, peft_config) -> None:
    super()._validate_peft_config(peft_config)
    if not self.config.load_vlm_weights:
        logging.warning(
            "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
            "Set `load_vlm_weights=True` to fine-tune the existing policy."
        )
```
- **목적**: PEFT 설정 검증 및 경고

---

## 4. VLAFlowMatching 클래스 (529-904줄)

### 클래스 정의 및 초기화 (529-600줄)

#### __init__ 메서드 (555-600줄)
```python
def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
    super().__init__()
    self.config = config
```

**주요 구성 요소**:

1. **VLM with Expert 모델** (559-570줄):
   ```python
   self.vlm_with_expert = SmolVLMWithExpertModel(
       model_id=self.config.vlm_model_name,  # 기본: "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
       freeze_vision_encoder=self.config.freeze_vision_encoder,  # Vision encoder 고정 여부
       train_expert_only=self.config.train_expert_only,  # Expert만 학습 여부
       load_vlm_weights=self.config.load_vlm_weights,  # VLM 가중치 로드 여부
       attention_mode=self.config.attention_mode,  # "cross_attn" 또는 "self_attn"
       num_expert_layers=self.config.num_expert_layers,  # Expert 레이어 수
       num_vlm_layers=self.config.num_vlm_layers,  # VLM 레이어 수
       self_attn_every_n_layers=self.config.self_attn_every_n_layers,  # Self-attention 간격
       expert_width_multiplier=self.config.expert_width_multiplier,  # Expert 너비 배수
       device=self.config.device if self.config.device is not None else "auto",
   )
   ```

2. **Projection 레이어들** (571-582줄):
   ```python
   # State → VLM hidden size
   self.state_proj = nn.Linear(
       self.config.max_state_dim, 
       self.vlm_with_expert.config.text_config.hidden_size
   )
   
   # Action → Expert hidden size
   self.action_in_proj = nn.Linear(
       self.config.max_action_dim, 
       self.vlm_with_expert.expert_hidden_size
   )
   
   # Expert hidden size → Action
   self.action_out_proj = nn.Linear(
       self.vlm_with_expert.expert_hidden_size, 
       self.config.max_action_dim
   )
   
   # Time + Action fusion MLP
   self.action_time_mlp_in = nn.Linear(
       self.vlm_with_expert.expert_hidden_size * 2,  # action + time
       self.vlm_with_expert.expert_hidden_size
   )
   self.action_time_mlp_out = nn.Linear(
       self.vlm_with_expert.expert_hidden_size,
       self.vlm_with_expert.expert_hidden_size
   )
   ```

3. **특수 토큰 설정** (584-593줄):
   ```python
   self.set_requires_grad()  # state_proj의 requires_grad 설정
   self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
   self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
   self.global_image_start_token = torch.tensor(
       [self.fake_image_token, self.global_image_token], dtype=torch.long
   )
   self.add_image_special_tokens = self.config.add_image_special_tokens
   self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
   self.prefix_length = self.config.prefix_length
   self.rtc_processor = rtc_processor
   ```

4. **모델 컴파일** (596-600줄):
   ```python
   if config.compile_model:
       torch.set_float32_matmul_precision("high")  # 높은 정밀도 설정
       self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
       self.forward = torch.compile(self.forward, mode=config.compile_mode)
   ```
   - **목적**: 추론 속도 향상 (PyTorch 2.0+)

### 유틸리티 메서드들 (602-623줄)

#### _rtc_enabled (602-603줄)
```python
def _rtc_enabled(self):
    return self.config.rtc_config is not None and self.config.rtc_config.enabled
```

#### set_requires_grad (605-607줄)
```python
def set_requires_grad(self):
    for params in self.state_proj.parameters():
        params.requires_grad = self.config.train_state_proj
```
- **목적**: state_proj의 학습 여부 제어

#### sample_noise (609-617줄)
```python
def sample_noise(self, shape, device):
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )
    return noise
```
- **목적**: Flow Matching을 위한 노이즈 샘플링

#### sample_time (619-623줄)
```python
def sample_time(self, bsize, device):
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
    time = time_beta * 0.999 + 0.001  # [0.001, 1.0] 범위로 제한
    return time
```
- **목적**: Flow Matching의 timestep 샘플링
- **Beta 분포**: 0에 가까운 값에 더 많은 샘플 (초기 denoising에 집중)

### embed_prefix 메서드 (625-717줄) - Prefix 임베딩
```python
def embed_prefix(
    self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """이미지, 언어, 상태를 임베딩하여 prefix 생성"""
```

**단계별 설명**:

1. **초기화** (631-633줄):
   ```python
   embs = []  # 임베딩 리스트
   pad_masks = []  # 패딩 마스크 리스트
   att_masks = []  # 어텐션 마스크 리스트
   ```

2. **이미지 임베딩** (634-680줄):
   ```python
   for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
       # 이미지 시작 토큰 (선택적)
       if self.add_image_special_tokens:
           image_start_token = self.vlm_with_expert.embed_language_tokens(
               self.global_image_start_token.to(device=...)
           ).unsqueeze(0).expand(img.shape[0], -1, -1)
           # ... 마스크 추가
       
       # 이미지 임베딩
       img_emb = self.vlm_with_expert.embed_image(img)
       
       # 정규화: sqrt(dim) 곱하기
       img_emb_dim = img_emb.shape[-1]
       img_emb = img_emb * torch.tensor(img_emb_dim**0.5, ...)
       
       # 이미지 끝 토큰 (선택적)
       if self.add_image_special_tokens:
           image_end_token = ...
   ```

3. **언어 임베딩** (681-690줄):
   ```python
   lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
   lang_emb = lang_emb * math.sqrt(lang_emb_dim)  # 정규화
   embs.append(lang_emb)
   pad_masks.append(lang_masks)
   att_masks += [0] * num_lang_embs  # 언어는 서로 볼 수 있음
   ```

4. **상태 임베딩** (692-703줄):
   ```python
   state_emb = self.state_proj(state)  # Linear projection
   state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
   embs.append(state_emb)
   pad_masks.append(state_mask)
   att_masks += [1] * (states_seq_len)  # 상태는 액션을 볼 수 없음
   ```

5. **결합 및 패딩** (704-717줄):
   ```python
   embs = torch.cat(embs, dim=1)  # 시퀀스 차원으로 결합
   pad_masks = torch.cat(pad_masks, dim=1)
   att_masks = torch.tensor(att_masks, ...)
   
   # prefix_length로 패딩
   if seq_len < self.prefix_length:
       embs = pad_tensor(embs, self.prefix_length, pad_value=0)
       pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
       att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
   
   att_masks = att_masks.expand(bsize, -1)
   ```

**반환값**:
- `embs`: 결합된 임베딩 (B, prefix_len, hidden_size)
- `pad_masks`: 패딩 마스크 (B, prefix_len)
- `att_masks`: 어텐션 마스크 (B, prefix_len)

### embed_suffix 메서드 (719-760줄) - Suffix 임베딩
```python
def embed_suffix(self, noisy_actions, timestep):
    """노이즈가 섞인 액션과 timestep을 임베딩"""
```

**단계별 설명**:

1. **액션 임베딩** (726줄):
   ```python
   action_emb = self.action_in_proj(noisy_actions)  # (B, chunk_size, expert_hidden_size)
   ```

2. **Timestep 임베딩** (730-738줄):
   ```python
   time_emb = create_sinusoidal_pos_embedding(
       timestep,  # (B,)
       self.vlm_with_expert.expert_hidden_size,
       self.config.min_period,
       self.config.max_period,
       device=device,
   )
   time_emb = time_emb.type(dtype=dtype)
   time_emb = time_emb[:, None, :].expand_as(action_emb)  # (B, chunk_size, expert_hidden_size)
   ```

3. **액션+타임 융합** (741-745줄):
   ```python
   action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # (B, chunk_size, 2*hidden_size)
   action_time_emb = self.action_time_mlp_in(action_time_emb)
   action_time_emb = F.silu(action_time_emb)  # Swish 활성화
   action_time_emb = self.action_time_mlp_out(action_time_emb)
   ```

4. **마스크 생성** (748-759줄):
   ```python
   embs.append(action_time_emb)
   pad_masks.append(action_time_mask)
   att_masks += [1] * self.config.chunk_size  # 액션은 prefix를 볼 수 없음
   ```

**반환값**: prefix와 동일한 형식

### forward 메서드 (762-798줄) - 학습용
```python
def forward(
    self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
) -> Tensor:
    """학습 시 loss 계산"""
```

**단계별 설명**:

1. **노이즈 및 timestep 샘플링** (766-770줄):
   ```python
   if noise is None:
       noise = self.sample_noise(actions.shape, actions.device)
   if time is None:
       time = self.sample_time(actions.shape[0], actions.device)
   ```

2. **Flow Matching 준비** (772-774줄):
   ```python
   time_expanded = time[:, None, None]  # (B, 1, 1)
   x_t = time_expanded * noise + (1 - time_expanded) * actions  # 노이즈와 액션 보간
   u_t = noise - actions  # 목표 벡터 필드
   ```
   - **x_t**: timestep t에서의 노이즈가 섞인 액션
   - **u_t**: 모델이 예측해야 하는 벡터 필드

3. **Prefix/Suffix 임베딩** (775-778줄):
   ```python
   prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
   suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
   ```

4. **마스크 결합** (780-783줄):
   ```python
   pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
   att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
   att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
   position_ids = torch.cumsum(pad_masks, dim=1) - 1
   ```

5. **모델 forward** (785-792줄):
   ```python
   (_, suffix_out), _ = self.vlm_with_expert.forward(
       attention_mask=att_2d_masks,
       position_ids=position_ids,
       past_key_values=None,  # 학습 시 캐시 사용 안 함
       inputs_embeds=[prefix_embs, suffix_embs],
       use_cache=False,
       fill_kv_cache=False,
   )
   ```

6. **Loss 계산** (793-798줄):
   ```python
   suffix_out = suffix_out[:, -self.config.chunk_size :]  # 마지막 chunk_size만 사용
   suffix_out = suffix_out.to(dtype=torch.float32)  # float32로 변환
   v_t = self.action_out_proj(suffix_out)  # 액션 예측
   losses = F.mse_loss(u_t, v_t, reduction="none")  # MSE loss
   return losses  # (B, chunk_size, action_dim)
   ```

### sample_actions 메서드 (800-869줄) - 추론용
```python
def sample_actions(
    self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
) -> Tensor:
    """추론 시 액션 생성"""
```

**단계별 설명**:

1. **초기화** (811-816줄):
   ```python
   bsize = state.shape[0]
   device = state.device
   if noise is None:
       actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
       noise = self.sample_noise(actions_shape, device)
   ```

2. **Prefix 임베딩 및 KV 캐시 생성** (818-831줄):
   ```python
   prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
   prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
   prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
   
   # KV 캐시 생성 (한 번만 계산)
   _, past_key_values = self.vlm_with_expert.forward(
       attention_mask=prefix_att_2d_masks,
       position_ids=prefix_position_ids,
       past_key_values=None,
       inputs_embeds=[prefix_embs, None],  # prefix만, suffix는 None
       use_cache=self.config.use_cache,
       fill_kv_cache=True,  # 캐시 채우기
   )
   ```

3. **Flow Matching 반복** (832-868줄):
   ```python
   num_steps = self.config.num_steps  # 기본: 10
   dt = -1.0 / num_steps  # 시간 스텝 크기
   
   x_t = noise  # 초기값: 순수 노이즈
   for step in range(num_steps):
       time = 1.0 + step * dt  # 1.0 → 0.0 (역방향)
       time_tensor = torch.tensor(time, ...).expand(bsize)
       
       # Denoising step
       def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
           return self.denoise_step(
               x_t=input_x_t,
               prefix_pad_masks=prefix_pad_masks,
               past_key_values=past_key_values,  # 재사용!
               timestep=current_timestep,
           )
       
       # RTC 처리 (선택적)
       if self._rtc_enabled():
           v_t = self.rtc_processor.denoise_step(...)
       else:
           v_t = denoise_step_partial_call(x_t)
       
       # Euler step: x_{t-1} = x_t + dt * v_t
       x_t = x_t + dt * v_t
       
       # 디버깅 (선택적)
       if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
           self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
   
   return x_t  # 최종 액션
   ```

**핵심 최적화**:
- Prefix의 KV 캐시를 한 번만 계산하고 재사용
- 각 denoising step에서는 suffix만 처리

### denoise_step 메서드 (871-904줄) - 단일 Denoising Step
```python
def denoise_step(
    self, prefix_pad_masks, past_key_values, x_t, timestep,
):
    """단일 denoising step 수행"""
```

**단계별 설명**:

1. **Suffix 임베딩** (879줄):
   ```python
   suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
   ```

2. **어텐션 마스크 구성** (881-888줄):
   ```python
   suffix_len = suffix_pad_masks.shape[1]
   batch_size = prefix_pad_masks.shape[0]
   prefix_len = prefix_pad_masks.shape[1]
   
   # Prefix는 suffix를 볼 수 있음 (2D 마스크)
   prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
   
   # Suffix 내부 어텐션 마스크
   suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
   
   # 결합
   full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
   ```

3. **Position IDs 계산** (889-890줄):
   ```python
   prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # prefix 길이
   position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
   ```

4. **모델 forward** (892-899줄):
   ```python
   outputs_embeds, _ = self.vlm_with_expert.forward(
       attention_mask=full_att_2d_masks,
       position_ids=position_ids,
       past_key_values=past_key_values,  # 재사용!
       inputs_embeds=[None, suffix_embs],  # prefix는 None (캐시 사용)
       use_cache=self.config.use_cache,
       fill_kv_cache=False,  # 이미 채워져 있음
   )
   ```

5. **액션 예측** (900-904줄):
   ```python
   suffix_out = outputs_embeds[1]  # suffix 출력만
   suffix_out = suffix_out[:, -self.config.chunk_size :]  # 마지막 chunk_size
   suffix_out = suffix_out.to(dtype=torch.float32)
   v_t = self.action_out_proj(suffix_out)  # 벡터 필드 예측
   return v_t
   ```

---

## 5. 전체 흐름 요약

### 학습 흐름
1. `forward()` 호출
2. 노이즈 및 timestep 샘플링
3. Flow Matching: `x_t = t * noise + (1-t) * actions`
4. Prefix/Suffix 임베딩
5. VLM+Expert forward
6. Loss 계산 (MSE)

### 추론 흐름
1. `sample_actions()` 호출
2. Prefix 임베딩 및 KV 캐시 생성
3. 노이즈로 시작 (`x_t = noise`)
4. 반복적으로 denoising:
   - `denoise_step()` 호출
   - `x_t = x_t + dt * v_t` (Euler step)
5. 최종 액션 반환

### 주요 최적화 기법
- **KV 캐싱**: Prefix는 한 번만 계산
- **큐 관리**: 액션을 미리 생성하여 효율성 향상
- **모델 컴파일**: `torch.compile`로 속도 향상
- **RTC**: 실시간 추론 시 연속성 유지

---

## 6. 핵심 개념 정리

### Flow Matching
- 확산 모델의 일반화
- 노이즈에서 액션으로의 경로를 학습
- `u_t = noise - actions`를 예측

### Prefix-Suffix 구조
- **Prefix**: 이미지, 언어, 상태 (고정)
- **Suffix**: 액션 (변화)
- Prefix는 Suffix를 볼 수 있지만, Suffix는 Prefix를 볼 수 없음

### Attention Mask 전략
- 이미지/언어: 서로 볼 수 있음 (att_mask=0)
- 상태: 액션을 볼 수 없음 (att_mask=1)
- 액션: Prefix를 볼 수 없음 (att_mask=1)

### Aloha 호환성
- 조인트 방향 반전
- 그리퍼 변환 (선형 ↔ 각도)
- 학습/추론 시 다른 변환 적용
