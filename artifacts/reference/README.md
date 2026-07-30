# artifacts/reference/ — 강사 사전 실행 결과

강사가 T4에서 1회 완주한 결과를 `artifacts/` 와 **같은 상대 경로**로 여기에 복사해 둔다.

```
artifacts/reference/runs/<run_id>/config.json
artifacts/reference/eval/<run_id>/summary.json
...
```

GPU 할당에 실패한 수강생은 `430`·`450`에서 `USE_REFERENCE = True` 로 두면
이 결과를 읽어 **분석·해석 활동에 참여**할 수 있다(학습은 못 하지만 평가는 배운다).

개편계획서 §7-1의 GPU 3중 백업 중 (A)에 해당한다.
