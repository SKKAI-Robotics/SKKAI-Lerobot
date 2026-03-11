# Aloha 변환 상세 설명

## 1. Aloha란?

**Aloha (A Low-cost Open-source Hardware for Autonomous manipulation)**는 로봇 조작을 위한 오픈소스 하드웨어 및 소프트웨어 프레임워크입니다.

- **목적**: 저비용으로 로봇 조작 학습 및 실행
- **특징**: 
  - 양팔 로봇 (dual-arm robot)
  - 각 팔에 7개의 관절 (7-DOF)
  - 그리퍼 포함
  - 총 14개 관절 + 2개 그리퍼 = 16차원 액션/상태 공간

## 2. 왜 변환이 필요한가?

SmolVLA는 **다른 형식**으로 액션과 상태를 표현합니다:

1. **조인트 방향**: 일부 조인트의 부호가 반대
2. **그리퍼 표현**: 
   - Aloha: **선형 공간** (linear space) - 그리퍼의 실제 물리적 위치
   - SmolVLA: **각도 공간** (angular space) - 그리퍼 조인트 각도

이 차이 때문에 Aloha 로봇에서 SmolVLA를 사용하려면 변환이 필요합니다.

---

## 3. 변환의 종류

### 3.1 조인트 방향 반전 (Joint Sign Flipping)

**문제**: Aloha와 SmolVLA가 일부 조인트의 양수/음수 방향을 반대로 정의

**해결**: 특정 모터 인덱스의 부호를 반전

```python
# 변환 대상 모터 인덱스
motor_indices_to_flip = [1, 2, 8, 9]

# Aloha → SmolVLA: 부호 반전
state[motor_idx] *= -1

# SmolVLA → Aloha: 다시 부호 반전 (원복)
actions[motor_idx] *= -1
```

**왜 이 모터들인가?**
- 로봇의 좌우 팔에서 특정 조인트의 좌표계 정의가 다름
- 인덱스 1, 2: 왼쪽 팔의 특정 조인트
- 인덱스 8, 9: 오른쪽 팔의 특정 조인트

### 3.2 그리퍼 변환 (Gripper Transformation)

**문제**: 
- Aloha는 그리퍼를 **선형 위치**로 표현 (예: 0.01844 ~ 0.05800 미터)
- SmolVLA는 그리퍼를 **각도**로 표현 (예: 0.4 ~ 1.5 라디안)

**해결**: 선형 ↔ 각도 변환

---

## 4. 그리퍼 변환 상세 분석

### 4.1 Aloha의 그리퍼 표현

Aloha는 그리퍼를 **물리적 선형 위치**로 표현합니다:

```python
# Aloha 그리퍼 값 범위 (정규화 전)
PUPPET_GRIPPER_POSITION_OPEN = 0.01844  # 열림 (미터)
PUPPET_GRIPPER_POSITION_CLOSED = 0.05800  # 닫힘 (미터)

# Aloha 그리퍼 조인트 값 (정규화 후)
PUPPET_GRIPPER_JOINT_OPEN = -0.6213  # 열림
PUPPET_GRIPPER_JOINT_CLOSE = 1.4910  # 닫힘
```

### 4.2 SmolVLA의 그리퍼 표현

SmolVLA는 그리퍼를 **각도**로 표현합니다:

```python
# SmolVLA 그리퍼 각도 범위 (정규화 전)
gripper_angular_min = 0.4  # 라디안
gripper_angular_max = 1.5  # 라디안

# 정규화 후: [0, 1] 범위
```

### 4.3 변환 함수들

#### `aloha_gripper_to_angular` (Aloha → SmolVLA)

```python
def aloha_gripper_to_angular(value):
    """
    Aloha의 선형 그리퍼 값을 SmolVLA의 각도 공간으로 변환
    
    단계:
    1. 정규화 해제: [0, 1] → [0.01844, 0.05800] (미터)
    2. 선형 → 각도 변환 (역변환)
    3. 정규화: [0.4, 1.5] 라디안 → [0, 1]
    """
    # 1단계: 정규화 해제
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)
    
    # 2단계: 선형 위치 → 각도 변환
    def linear_to_radian(linear_position, arm_length, horn_radius):
        # Interbotix 코드의 역변환
        # 기하학적 변환 공식 사용
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / 
                (2 * horn_radius * linear_position)
        return safe_arcsin(value)  # [-1, 1] 범위로 클램핑 후 arcsin
    
    # Interbotix 하드웨어 상수
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    
    # 3단계: 정규화
    # 실제 로봇에서 측정된 값
    return normalize(value, min_val=0.4, max_val=1.5)
```

**변환 과정**:
```
Aloha 정규화 값 [0, 1]
    ↓ unnormalize(0.01844, 0.05800)
물리적 선형 위치 [0.01844, 0.05800] 미터
    ↓ linear_to_radian (기하학적 변환)
각도 [0.4, 1.5] 라디안
    ↓ normalize(0.4, 1.5)
SmolVLA 정규화 값 [0, 1]
```

#### `aloha_gripper_from_angular` (SmolVLA → Aloha)

```python
def aloha_gripper_from_angular(value):
    """
    SmolVLA의 각도 값을 Aloha 형식으로 변환
    
    단계:
    1. 정규화 해제: [0, 1] → [0.4, 1.5] 라디안
    2. Aloha 조인트 범위로 정규화: [-0.6213, 1.4910]
    """
    # 1단계: 정규화 해제
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    
    # 2단계: Aloha 조인트 범위로 정규화
    return normalize(value, min_val=-0.6213, max_val=1.4910)
```

**변환 과정**:
```
SmolVLA 정규화 값 [0, 1]
    ↓ unnormalize(0.4, 1.5)
각도 [0.4, 1.5] 라디안
    ↓ normalize(-0.6213, 1.4910)
Aloha 조인트 값 [-0.6213, 1.4910]
```

#### `aloha_gripper_from_angular_inv` (역변환)

```python
def aloha_gripper_from_angular_inv(value):
    """
    aloha_gripper_from_angular의 역변환
    학습 시 사용 (Aloha 데이터 → SmolVLA 형식)
    """
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)
```

---

## 5. 실제 사용 예시

### 5.1 학습 시 (Training)

```python
def forward(self, batch, ...):
    if self.config.adapt_to_pi_aloha:
        # 1. 상태 변환: Aloha → SmolVLA
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        #   - 조인트 [1,2,8,9] 부호 반전
        #   - 그리퍼 [6,13] 선형 → 각도
        
        # 2. 액션 변환: Aloha → SmolVLA
        batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        #   - 조인트 [1,2,8,9] 부호 반전
        #   - 그리퍼 [6,13] Aloha 조인트 → SmolVLA 각도
    
    # 모델 학습 (SmolVLA 형식으로)
    loss = self.model.forward(...)
```

**흐름**:
```
Aloha 데이터 (환경에서 수집)
    ↓ _pi_aloha_decode_state
    ↓ _pi_aloha_encode_actions_inv
SmolVLA 형식 (모델 학습)
```

### 5.2 추론 시 (Inference)

```python
def _get_action_chunk(self, batch, ...):
    # 모델이 SmolVLA 형식으로 액션 생성
    actions = self.model.sample_actions(...)
    
    if self.config.adapt_to_pi_aloha:
        # 액션 변환: SmolVLA → Aloha
        actions = self._pi_aloha_encode_actions(actions)
        #   - 조인트 [1,2,8,9] 부호 반전
        #   - 그리퍼 [6,13] SmolVLA 각도 → Aloha 조인트
    
    return actions  # Aloha 형식으로 반환
```

**흐름**:
```
SmolVLA 모델 (각도 공간)
    ↓ _pi_aloha_encode_actions
Aloha 형식 (로봇 실행)
```

### 5.3 상태 관찰 시

```python
def _prepare_batch(self, batch):
    if self.config.adapt_to_pi_aloha:
        # 상태 변환: Aloha → SmolVLA
        batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
    return batch
```

---

## 6. 변환 메서드 상세

### 6.1 `_pi_aloha_decode_state` (상태: Aloha → SmolVLA)

```python
def _pi_aloha_decode_state(self, state):
    """
    Aloha 상태를 SmolVLA 형식으로 변환
    
    Args:
        state: (batch_size, state_dim) - Aloha 형식 상태
    
    Returns:
        state: (batch_size, state_dim) - SmolVLA 형식 상태
    """
    # 1. 조인트 방향 반전
    for motor_idx in [1, 2, 8, 9]:
        state[:, motor_idx] *= -1
    
    # 2. 그리퍼 변환: 선형 → 각도
    for motor_idx in [6, 13]:  # 왼쪽/오른쪽 그리퍼
        state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
    
    return state
```

**변환 대상**:
- **조인트**: 인덱스 1, 2 (왼쪽 팔), 8, 9 (오른쪽 팔)
- **그리퍼**: 인덱스 6 (왼쪽), 13 (오른쪽)

### 6.2 `_pi_aloha_encode_actions` (액션: SmolVLA → Aloha)

```python
def _pi_aloha_encode_actions(self, actions):
    """
    SmolVLA 액션을 Aloha 형식으로 변환 (추론 시)
    
    Args:
        actions: (batch_size, n_steps, action_dim) - SmolVLA 형식
    
    Returns:
        actions: (batch_size, n_steps, action_dim) - Aloha 형식
    """
    # 1. 조인트 방향 반전
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    
    # 2. 그리퍼 변환: 각도 → Aloha 조인트
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular(
            actions[:, :, motor_idx]
        )
    
    return actions
```

### 6.3 `_pi_aloha_encode_actions_inv` (액션: Aloha → SmolVLA)

```python
def _pi_aloha_encode_actions_inv(self, actions):
    """
    Aloha 액션을 SmolVLA 형식으로 변환 (학습 시)
    
    Args:
        actions: (batch_size, n_steps, action_dim) - Aloha 형식
    
    Returns:
        actions: (batch_size, n_steps, action_dim) - SmolVLA 형식
    """
    # 1. 조인트 방향 반전
    for motor_idx in [1, 2, 8, 9]:
        actions[:, :, motor_idx] *= -1
    
    # 2. 그리퍼 변환: Aloha 조인트 → 각도
    for motor_idx in [6, 13]:
        actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(
            actions[:, :, motor_idx]
        )
    
    return actions
```

---

## 7. 왜 이런 변환이 필요한가?

### 7.1 하드웨어 차이

- **Aloha**: 실제 물리적 로봇 하드웨어의 좌표계 사용
- **SmolVLA**: 학습 시 사용한 내부 좌표계 사용

### 7.2 학습 데이터 차이

SmolVLA는 **다른 형식의 데이터**로 사전 학습되었을 수 있습니다:
- 다른 로봇 플랫폼
- 다른 좌표계 정의
- 다른 그리퍼 표현 방식

### 7.3 호환성 유지

`adapt_to_pi_aloha=True`로 설정하면:
- Aloha 로봇에서 수집한 데이터로 학습 가능
- Aloha 로봇에서 직접 실행 가능
- 변환 없이도 다른 로봇에서 사용 가능

---

## 8. 설정 방법

### 8.1 Config 설정

```python
config = SmolVLAConfig(
    adapt_to_pi_aloha=True,  # Aloha 변환 활성화
    # ... 기타 설정
)
```

### 8.2 사용 예시

```python
# Aloha 데이터로 학습
policy = SmolVLAPolicy(config)
policy.train()  # 자동으로 변환 적용

# Aloha 로봇에서 실행
action = policy.select_action(observation)  # 자동으로 변환 적용
```

---

## 9. 주의사항

### 9.1 모터 인덱스

- **조인트**: [1, 2, 8, 9] - 특정 조인트만 변환
- **그리퍼**: [6, 13] - 왼쪽/오른쪽 그리퍼

### 9.2 그리퍼 변환의 복잡성

그리퍼 변환은 **기하학적 변환**을 포함합니다:
- 선형 위치 ↔ 각도 변환
- 하드웨어 상수 사용 (arm_length, horn_radius)
- 실제 로봇에서 측정된 값 사용

### 9.3 정확도

변환은 **근사치**일 수 있습니다:
- 하드웨어 제조 차이
- 캘리브레이션 오차
- 마모 및 변형

---

## 10. 요약

### Aloha 변환이란?

**Aloha 로봇과 SmolVLA 모델 간의 액션/상태 표현 차이를 해결하는 변환**

### 주요 변환:

1. **조인트 방향 반전**: 특정 조인트 [1,2,8,9]의 부호 반전
2. **그리퍼 변환**: 
   - Aloha (선형) ↔ SmolVLA (각도)
   - 기하학적 변환 포함

### 사용 시점:

- **학습 시**: Aloha 데이터 → SmolVLA 형식
- **추론 시**: SmolVLA 액션 → Aloha 형식

### 설정:

```python
config.adapt_to_pi_aloha = True  # 활성화
```

이 변환을 통해 **Aloha 로봇에서 SmolVLA를 직접 사용**할 수 있습니다!
