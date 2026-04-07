# SmolVLA LIBERO Eval Conversation Report

작성일: 2026-04-07

## 1) 대상 평가 런

- Eval output dir: /root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_full_20260406T092014Z
- Eval result file: /root/workspace/andycho/IL/lerobot/outputs/eval/smolvla_libero_full_20260406T092014Z/eval_info.json
- Run log file: /root/workspace/andycho/IL/lerobot/nohup_smolvla_libero_full_20260406T092014Z.log

## 2) 평가 중단 여부

결론: 중간 중단이 아니라 완료된 런으로 판단됨.

근거:
- eval_info.json에 per_group, overall 집계가 모두 존재
- overall.n_episodes = 400으로 전체 집계 완료
- 현재 동일 eval 프로세스는 실행 중이 아님

참고:
- videos 디렉터리 파일 수는 0
- eval_info.json의 video_paths도 비어 있어, 이번 런은 비디오 저장 없이 수치 집계 중심으로 수행됨

## 3) 성능 요약

전체:
- pc_success: 69.5
- avg_sum_reward: 0.695
- avg_max_reward: 0.695
- n_episodes: 400

그룹별:
- libero_object: 90.0 (100 에피소드 중 90 성공)
- libero_goal: 74.0 (100 중 74)
- libero_spatial: 73.0 (100 중 73)
- libero_10: 41.0 (100 중 41)

구성:
- per_task 엔트리: 40개
- task당 시도 횟수: 10회
- 총 에피소드: 40 x 10 = 400

## 4) 시간 정보

- eval_s: 62823.5061초
- 시분초 환산: 약 17시간 27분 3초
- eval_ep_s: 157.0588초 (에피소드당 약 2분 37초)
- 결과 파일 modified(UTC): 2026-04-07 02:49:02
- 런 폴더명 타임스탬프: 20260406T092014Z

해석:
- 대략 2026-04-06 09:20 UTC 시작, 2026-04-07 02:49 UTC 종료로 일치

## 5) 체크포인트 출처와 파인튜닝 여부

결론:
- 체크포인트는 Hugging Face Hub에서 로드됨
- 사용 정책 경로: HuggingFaceVLA/smolvla_libero

근거:
- 프로토콜 문서의 실행 커맨드에 --policy.path=HuggingFaceVLA/smolvla_libero
- 로그에 huggingface.co/HuggingFaceVLA/smolvla_libero 요청 기록
- 로그 설정에 pretrained_path='HuggingFaceVLA/smolvla_libero'

LIBERO 파인튜닝 관련:
- 모델 계보 태그상 base_model:finetune:lerobot/smolvla_base 표기는 확인됨
- 다만 파인튜닝 강도(정확한 steps/epochs/샘플 수)는 공개 메타데이터만으로 확정 불가
- 허브 메타데이터에는 datasets: unknown 표기
- 파일 트리 기준 train_config.json 등 학습 상세 파일은 확인되지 않음

## 6) 모델 파라미터 크기 관련 정리

질문: 0.45B가 맞는가?

결론:
- SmolVLA는 일반적으로 450M(0.45B)급으로 안내됨

혼동 포인트:
- 런 로그의 VLM backbone 이름은 SmolVLM2-500M-Instruct로 보임
- 하지만 SmolVLA 정책은 백본 단순 사용이 아니라 구조적 경량화/설정이 함께 반영되어 450M급으로 설명됨

## 7) Rollout 설정 점검 결과

### 7-1) 한 에피소드 최대 스텝(max_steps)

결론:
- eval 루프는 env._max_episode_steps를 max_steps로 사용
- 이번 런은 env.episode_length=None으로 실행
- LIBERO는 episode_length가 None이면 suite별 기본 상한을 사용

LIBERO suite별 기본 상한:
- libero_spatial: 220
- libero_object: 280
- libero_goal: 300
- libero_10: 520
- libero_90: 400

해석:
- 기본값이 각 suite의 longest demo보다 약간 크게 잡혀 있어, 상한이 지나치게 짧아 조기 종료되는 설정은 아님

### 7-2) 액션 청크 실행 방식

결론:
- 현재 eval은 매 스텝 1개 action 실행 후 재관측하는 방식
- 청크 전체를 한 번에 open-loop로 밀어 넣는 방식이 아님

근거:
- eval 루프에서 매 반복 policy.select_action 호출 후 env.step(action) 1회 수행
- SmolVLA select_action은 내부 큐를 쓰더라도 반환은 popleft()로 단일 action
- 이번 실행 로그에서 n_action_steps=1

## 8) 시간 추정 관련 논의

- 현재 측정치(157초/에피소드) 기준으로 600 에피소드 예상 시간은 약 26.2시간
- 34시간은 초기화 지연, 병목, 리소스 변동이 큰 경우의 보수적 상한으로 볼 수 있음

## 9) 핵심 결론 요약

- 평가는 중간 중단이 아니라 정상 완료됨
- 성능은 전체 69.5%, 그룹별로 object가 높고 libero_10이 낮음
- 체크포인트는 HF의 HuggingFaceVLA/smolvla_libero 사용
- LIBERO 파인튜닝 계열 모델로 보되, 파인튜닝 강도 수치는 공개 정보로 확정 불가
- rollout 동작은 1-step 실행 후 재관측 루프이며, 이번 런은 n_action_steps=1

## 10) 후속 권장 점검

- 재현 비교 시 아래를 고정해서 A/B 테스트 권장
  - env.episode_length
  - seed
  - n_action_steps
  - use_async_envs
  - LIBERO assets 및 패키지 버전
- 목표가 평균 성능 향상인지, 추정 오차 축소인지 분리해서 rollout(에피소드 수) 확대 결정 권장
