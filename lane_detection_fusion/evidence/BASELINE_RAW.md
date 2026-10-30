# BAG_RAW: 원본 흰선 모드의 독립 오프라인 기준 결과

이번 실행은 새 `inputs/bags_raw/bag_raw/{normal,reverse}`만 사용했다. 이전640×480 bag의 통계는 사용하지 않았다.

## 입력 재구성과 실행 범위

기록에는 `/camera_raw_16_9`가 있고 `/camera`는 없다. native1280×720 JPEG를 디코딩해 camera_ros publisher와 같은 기하 순서인 `getOptimalNewCameraMatrix(alpha=0)` → native remap → validROI crop →640×480 resize를 적용하고 JPEG로 다시 인코딩했다. 이를 변경하지 않은 원본 `LaneDetection` callback에 입력했다.

**이것은 실제 `/camera`의 비트 정확한 복원이 아니다.** 실제 publisher는 JPEG 인코딩 전의 카메라 프레임에서 `/camera`를 만들지만 여기서는 이미 손실 압축된 raw JPEG에서 출발하므로 추가 JPEG 손실이 있다. 현재 분석 환경의 OpenCV와 원래 장비 버전 차이도 있다.

ROS imports·Node·publisher와 GUI만 가짜 객체로 대체했다. 원본 인식·제어 함수는 그대로 실행했다. mode=False(white), state=True, intersection=False를 명시했으며 기록 당시의 상태를 복구했다고 주장하지 않는다. ROS executor·전송·큐·타이머·실차 폐루프 재현은 아니다.

## 실제 처리 횟수

| BAG_RAW | 전체 카메라 프레임 | center=640 fallback | fallback 중 새 명령 계산 | 새 명령 없음(return) | 예외 |
|---|---:|---:|---:|---:|---:|
| normal | 2755 | 531 | 531 | 0 | 0 |
| reverse | 2642 | 525 | 502 | 34 | 0 |

| 확인한 연속 구간: 첫camera 기록 기준 초 | 프레임 | fallback | 새 명령 없음 |
|---|---:|---:|---:|
| normal 140.0–155.0초 | 193 | 12 | 0 |
| reverse 64.0–92.0초 | 347 | 115 | 0 |

원본은 유효 성분을 얻지 못하면 white 중심값을640으로 대체한다. 별도의 차선 없음 상태가 아니므로 이후 분기가 막지 않으면 이 가상 중심값으로도 선속도·각속도를 계산한다. `many_white` 조건으로 callback이 일찍 return하면 그 프레임에는 새 명령이 발행되지 않는다. 실제 하위 구동부가 이전 명령을 언제까지 유지하는지는 이 실행으로 확인하지 않았다.

## 해석 제한

- fallback·return 횟수는 코드 경로의 관측값이다. 독립 수동 정답이 없으므로 인식 정확도·오답률·주행 실패율로 부르지 않는다.
- 이 입력도 수동 주행 기록이며, 새 계산 명령이 저장된 미래 영상을 바꾸지 않는다. 완주·실제 복귀·직진 진동 개선을 증명하지 않는다.
- fusion 경로는 전체1280×720 rectification 후640×360을 유지한다. 이 baseline은 기존 publisher의 ROI crop과640×480 변형을 재구성한다. **동일 픽셀·동일 시야·동일 좌표계의 직접 수치 비교가 아니다.** callback 시간도 분석 PC 계측이며 로봇 끝단 지연이 아니다.
- 표의 시간은 각 bag 첫camera `messages.timestamp`를0초로 했다. 센서 header와 기록시각의 차이를 실기 지연으로 해석하지 않는다.

## 재현 자료

- `evidence/baseline_raw/{normal,reverse}/frames.csv`: 전 프레임·기록/센서 시각·fallback·mask 관측·새 계산 명령·return·예외.
- 같은 폴더의 `summary.json`: 전 프레임과 요청한 연속 구간의 집계, 출처·실행 가정.
- `evidence/baseline_raw/manifest.json`, `source_snapshot.py`: 원본 코드·설정·도구 해시와 분석 환경.
- `tools/compare_baseline_raw.py`로 같은 BAG_RAW 입력에서 재실행할 수 있다.

### ZIP을 다른 분석 PC에 옮겼을 때

이 기준 실행 도구에는 **NumPy, OpenCV(`cv2`), `rosbags`**가 필요하다. ROS 설치와 실제 카메라는 필요하지 않으며, 로봇용 노드가 아닌 별도 분석 도구다. 이번 실행은 manifest에 기록한 Python3.12.14 환경에서 확인했다. 설치할 패키지의 해당 Python 지원 범위를 확인한다.

```bash
python -m pip install numpy opencv-python rosbags
python tools/compare_baseline_raw.py --bag-root /path/to/bag_raw
```

`bag_raw` 아래에는 `normal/normal_0.db3`, `reverse/reverse_0.db3`가 있어야 한다. bag 자체는 별도로 준비한다. 원래 workspace의 source/helper가 없으면 ZIP에 포함한 `evidence/baseline_raw/source_snapshot.py`와 `replay_helper.py`를 자동 선택한다. 항상 보존 스냅샷을 명시하려면 다음과 같이 실행한다.

```bash
python tools/compare_baseline_raw.py --bag-root /path/to/bag_raw --source evidence/baseline_raw/source_snapshot.py --helper evidence/baseline_raw/replay_helper.py
```

결과는 `evidence/baseline_raw/`에 다시 기록되므로 기존 결과를 보존하려면 분석용 ZIP 사본에서 실행한다. 스냅샷 해시와 원래 실행 manifest의 일치는 `portability_addendum`에 기록했다. 이식용 기본 경로·설명만 추가했으며 위 전수치 계산은 재실행하지 않았다.
