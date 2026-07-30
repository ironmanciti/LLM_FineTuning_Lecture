# LLM 작동 원리와 오픈웨이트 모델 파인튜닝 — 실습 저장소

24시간(3일 × 8교시) 집체 훈련과정 실습 코드. 개편계획서
`개편계획/LLM 작동 원리와 오픈웨이트 모델 파인튜닝.md` 의 §5에 대응한다.

> **상태: 초안.** 노트북 로직과 공용 모듈은 CPU 구간까지 검증했으나
> **GPU 구간(300·420·430·440)은 T4에서 1회 완주 검증이 필요하다**(§강사 준비 1~2).
> 검증 전에 수업에 투입하지 마십시오.

---

## 1. 진행 순서

| 노트북 | 배치 | GPU | 하는 일 | 산출물 |
|---|---|:--:|---|---|
| `010_Tokenizers` | 1일차 3교시 | — | 형태소 vs subword, BPE 병합, 특수 토큰 | — |
| `020_Korean_Token_Cost` | 1일차 4교시 | — | fertility 측정 → **`max_seq_len` 권고값 산출** | `tokens/token_cost.json` |
| `300_Open_Weight_Anatomy` | 1일차 8교시 | ✔ | fp16 vs 4bit VRAM, config 대조, 로짓 분포, KV 캐시 지연 | `anatomy.json` |
| `410_SFT_Dataset_Build` | 2일차 8교시 | — | 정제 → chat template → **손실 마스킹 검증** → 품질 5항목 | `data/*.jsonl` |
| `420_QLoRA_SFT_Training` | **3일차 2교시** | ✔ | QLoRA 학습 실행 | `runs/<run_id>/` |
| `430_Evaluate_Before_After` | 3일차 6교시 | ✔ | 전후 동일 프로토콜 채점 + **회귀 점검** | `eval/<run_id>/` |
| `440_Merge_And_ModelCard` | 3일차 7교시 | ✔ | 병합 → 재평가 → 모델 카드 | `merged/<run_id>/` |
| `450_Final_Report` | 3일차 8교시 | — | 전 결과 집계 → 최종 리포트 | `report/<run_id>/` |

**시간 배치 주의**: `420`은 3일차 **2교시에 학습을 걸어두고** 3교시(진단 강의) 동안 돌린 뒤
4교시에 결과를 본다. 학습이 끝나기를 기다리며 강의를 멈추면 시간이 맞지 않는다.

---

## 2. 데이터 계약 — `artifacts/`

노트북은 서로를 직접 호출하지 않는다. 정해진 경로에 쓰고 읽는다.
**경로를 바꾸려면 `common/artifacts.py` 한 곳만 고친다.**

```
artifacts/
  env.json                          <- 300
  anatomy.json                      <- 300
  tokens/token_cost.json            <- 020   (recommended_max_seq_len)
  data/train.jsonl                  <- 410 -> 420
  data/eval.jsonl                   <- 410 -> 430·440
  data/domain30.jsonl               <- 410 -> 430   (수강생 직접 작성)
  data/data_report.json             <- 410   (품질 점검·누수 검증)
  runs/<run_id>/config.json          <- 420 -> 440·450
  runs/<run_id>/history.json         <- 420 -> 450
  runs/<run_id>/loss_curve.png       <- 420
  runs/<run_id>/adapter/             <- 420 -> 430·440
  eval/<run_id>/<cond>/metrics.json  <- 430·440   cond = base | tuned | merged
  eval/<run_id>/<cond>/preds.jsonl   <- 430·440
  eval/<run_id>/summary.json         <- 430 -> 440·450
  eval/<run_id>/rubric.json          <- 430 -> 450
  eval/<run_id>/regression.json      <- 430 -> 450
  merged/<run_id>/model_card.md      <- 440
  report/<run_id>/final_report.md    <- 450
  report/<run_id>/comparison.csv     <- 450
  reference/                        <- 강사 사전 실행 결과 (GPU 실패 시 대체 입력)
```

`artifacts/`는 `.gitignore` 대상이다. 산출물은 각자 로컬에 쌓인다.

---

## 3. 공용 모듈 — `common/`

| 파일 | 역할 | 관련 슬라이드 |
|---|---|---|
| `config.py` | **모델명·데이터·하이퍼파라미터 단일 지점.** 개강일에 여기만 고친다 | — |
| `env.py` | seed 고정 / 환경 진단(**bf16 지원 여부**) / VRAM 측정 / KV 캐시 계산 | 1일차 §3.2-4, §3.3-4 |
| `chat.py` | chat template 적용 / **손실 마스킹 + 색상 렌더링** / 자동 점검 | 2일차 §3.9-3·4 |
| `metrics.py` | EM·문자 F1 / **형식 준수율** / 부트스트랩 CI / 혼동행렬·macro-F1 | 3일차 §3.12-1~5 |
| `regression.py` | 회귀 세트 채점 / 학습 전후 비교 / 퇴화 탐지 | 3일차 §3.12-8 |
| `artifacts.py` | 위 데이터 계약 | — |

**지표 함수를 노트북마다 다시 쓰지 말 것.** 조건별로 다른 코드로 채점하면 비교가 성립하지 않는다.

---

## 4. 설치

### 로컬 (권장 — VS Code + AI 코딩 에이전트 + uv + Git)
```bash
git clone <저장소 주소>
cd LLM_FineTuning_Lecture
uv venv && uv pip install -r requirements.txt
```
KoNLPy(`010`)는 JDK가 필요하다. 없으면 해당 셀만 건너뛰어도 실습이 성립한다.

### Colab (GPU 실습)
각 노트북 첫 셀의 `INSTALL = True` 로 바꾸고 1회 실행. 설치 후 런타임 재시작.
한글 폰트: `!apt-get install -y fonts-nanum` 후 런타임 재시작.

---

## 5. T4 제약 — 반드시 알아야 할 3가지

1. **bf16 미지원(Turing).** `env.pick_dtype()`이 자동으로 fp16으로 내려간다.
   fp16 오버플로로 loss가 NaN이 될 수 있다 → LR 절반, `max_grad_norm` 확인.
2. **FlashAttention 미지원.** `attn_implementation="sdpa"`를 사용한다.
3. **16GB.** 0.5B가 기본, 1.5B는 `max_seq_len`을 줄이면 가능, 3B는 선택 확장.
   `common/config.py`의 `MODEL_CANDIDATES`에서 전환한다.

---

## 6. 강사 준비 체크리스트

개강 2주 전:

- [ ] **T4에서 `010` → `450` 전량 1회 완주.** 이 저장소는 아직 GPU 검증 전이다.
- [ ] **T4 실측표 작성** — 0.5B/1.5B × `max_seq_len` 512·1024 × batch 1·2·4의
      VRAM 피크와 학습 시간. 개편계획서 §3.6-7·§3.8-4·§3.10의 슬라이드 표를 이 값으로 채운다.
      **이론 계산과 20~30% 차이가 난다.**
- [ ] **`420` 학습 시간 실측** → 3일차 2교시 배치가 성립하는지 확인.
      성립하지 않으면 `config.N_TRAIN`을 줄인다.
- [ ] **fp16 NaN 1회 재현** → 스크린샷 확보 (3일차 §3.11-3 슬라이드용).
- [ ] **`data/regression_set.jsonl` 편집** — `data/README.md`의 체크리스트 참조.
      베이스 모델이 20문항 중 15개 이상 통과해야 회귀 지표가 유효하다.
- [ ] **모델·토크나이저 오프라인 미러** — 20명 동시 다운로드는 실패한다고 가정.
      인스트럭트 sLLM 2종 + 토크나이저 5종 ≈ 4~6GB.
- [ ] **`artifacts/reference/` 채우기** — 자기 실행 결과를 넣어 두면, GPU 할당에
      실패한 수강생이 `USE_REFERENCE=True`로 분석 활동에 참여할 수 있다.
- [ ] **gated 모델 접근 확인** — `config.TOKENIZER_ZOO`의 Llama·Gemma는 HF 토큰이 필요하다.
      막히면 해당 항목을 제거해도 `020`이 성립한다.
- [ ] **`config.MODEL_ID` 라이선스 재확인** — 2일차 §3.6-4의 판정 실습 대상이다.

---

## 7. 설계 의도 — 왜 태스크가 JSON 출력인가

0.5B급 소형 모델에서 **EM/F1은 잘 오르지 않지만 형식 준수율은 뚜렷하게 오른다.**
그래서 SFT 태스크를 `{"answer","evidence"}` JSON 출력으로 설계했다.

파인튜닝의 효과를 **정직하게 관측할 수 있는 지표를 태스크 설계에 미리 심어 둔 것**이다.
3일차에 EM/F1이 오르지 않는 것은 실습 실패가 아니라 예정된 관찰이며, 강의자료
슬라이드 「LoRA, 왜 실무에서 기대만큼 안 되는가?」의 메시지와 정확히 일치한다.

강사가 이 의도를 사전에 내재화하지 않으면 수업이 "실습 실패"로 읽힌다.

---

## 8. 폐지·이관된 구 노트북

| 구 노트북 | 조치 |
|---|---|
| `040-TFIDF_Embedding`, `070-Project_Movie_Review_TFIDF` | → 「머신러닝 실무 입문」 이관 |
| `050-Data_Preprocessing` | → `410`의 정제 단계로 흡수 |
| `300-HuggingFace Pipeline`, `350_autoregressive_language_generation` | → `300`으로 통합 |
| `320-HuggingFace Sentiment Fine Tuning` | → 「Python과 Pytorch 딥러닝 입문」 일원화 |
| `400_LoRA_KorQuAD_Finetuning` | → `420`으로 확장·개명 |
| `600/601 Text 생성 및 Prompt` | → 「LLM 애플리케이션 개발」 이관 |
| `700-Project_Movie_감성분석` | → `450`에 흡수 |
| `710/712 RAG` | → 「RAG 시스템 구축과 평가」 이관 |
"# LLM_FineTuning_Lecture" 
