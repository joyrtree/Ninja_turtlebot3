# TurtleBot 차선 추적·제어 후보 v0.1

2026-10-30 회의용 구현. 기존 저장소를 수정하지 않은 별도 ROS 2 Python 패키지다.
**새 BAG_RAW만** 재처리에 사용한다. 이전 normal/unnormal 기록의 수치는 이번 검증에 포함하지 않는다.
최종 실행 결과와 남은 실패는 `evidence/MEETING.md`에 정리한다.

## 이번에 바꾼 논리

| 문제 | 구현한 처리 | 남은 한계 |
| --- | --- | --- |
| 노란 선이 반사로 흰색처럼 보임 | 밝기 대비와 선의 연속성을 사용하고 색은 약한 후보 단서로만 사용 | 가늘고 길게 생긴 반사는 여전히 선과 혼동 가능 |
| 따라가던 선 대신 반대쪽 선이 나타남 | 좌·우 경계의 시간상 대응을 유지. 두 선이 확인될 때 행별 폭을 갱신하고, 한 선만 남으면 최근 폭으로 중심 복원 | 초기 한 선만으로 좌우를 확정하지 않음. 두 선이 보이는 위치에서 초기화 필요 |
| 반폭을 더할지 뺄지 혼동 | 왼쪽 경계에는 폭/2를 더하고 오른쪽에는 빼서 같은 중심 생성. 영상 절반·색으로 역할을 바꾸지 않음 | 급격한 시점 변화로 대응이 끊기면 정지 후 재초기화 |
| 가까이/멀리 보는 딜레마 | 여러 높이에서 경계 곡선을 연결. 가까운 중심 오차와 더 먼 경로 방향을 별도 반영 | 영상 높이는 미터 단위 추종 거리가 아님. 카메라 장착 자세 변화 시 재확인 |
| 선 소실 후 기존 속도·조향 유지 | `lost/ambiguous/invalid`면 선속도·각속도 모두 0 | 정지 비율이 높으면 검출·기하 조건을 개선해야 함 |
| 커브·한 선 구간에서 조향이 못 따라옴 | 조향 요구량·영상 굽음·관측 품질과 한 선 상태에 따라 속도 명령 감소 | 실제 조향·구동 응답을 측정하지 않아 주행 효과 미검증 |
| 처리 밀림·늦은 명령 | 입력 대기 1장 교체, 추론 작업 분리, 처리 전후 만료 검사, 독립 정지 감시 | 로봇에서 카메라→구동기 전체 지연은 별도 측정 필요 |

원본을 무조건 복잡한 PID로 교체하지 않았다. 영상 정규화 좌우 오차 P + 경로 방향 항이며 I는 없고 D 기본값도 0이다.
지면 좌표 보정이 없어 **미터 기반 Pure Pursuit/MPC로 부르지 않는다**.

## 카메라 보정 연결

제공된 `camera_ros-main.zip`의 실제 보정 파일을 숫자 JSON으로 변환했다.
원본 pickle SHA-256: `c22cd925234e39a8fa587a9d9076f73fb4edcfb2b8e8110b406568a3f75a1da8`.
원본 보정은 1280×720, 유효 촬영 43장, RMS 1.617815이다. RMS는 실제 차선 인식 정확도가 아니다.

- 기본 입력: `/camera_raw_16_9`, `sensor_msgs/msg/CompressedImage`, 원본 1280×720.
- `raw_16_9`: alpha=0 왜곡 보정 맵을 한 번 만들고 재사용 → 640×360. 출력 캔버스에서 추가 ROI 크롭·4:3 변형을 하지 않는다. alpha=0 자체는 유효 화각을 바꾼다.
- 이미 보정된 `/camera`를 쓸 경우 프로필의 `input_mode`를 **`rectified`**로 바꾸고 입력 토픽도 바꾼다. 이 모드에서는 재보정하지 않는다.
- 네이티브 보정과 다른 크기를 raw 모드에 넣으면 오류 처리한다. 카메라 장착 자세·렌즈 설정이 바뀌면 유효성을 다시 확인한다.
- 출처와 처리 영상은 `evidence/camera_provenance.json`에 있다. 지면 외부 보정·차선의 실제 폭은 제공되지 않았다.

## 로봇에 설치

아래는 ROS 2가 설치된 로봇에서 실행할 명령이다. 이 PC에서는 ROS가 없어 실제 ROS 빌드는 확인하지 못했다.
선배 카메라 문서의 대상은 Galactic/Python 3.8/OpenCV 4.2이며, 로컬 검증은 Python 3.12/OpenCV 5.0에서 수행했다.

```bash
# 이 폴더를 ROS 작업공간 src/lane_detection_fusion으로 복사한 뒤
cd ~/ros2_ws
rosdep install --from-paths src/lane_detection_fusion --ignore-src -r -y
colcon build --packages-select lane_detection_fusion
source install/setup.bash
ros2 run lane_detection_fusion fusion --ros-args -p debug_enabled:=true
```

기본 출력은 바퀴에 연결하지 않는 `/lane_fusion/cmd_vel_preview`다. 초기 `/lane_state`는 false다.
카메라가 켜진 상태에서 미리보기 계산을 활성화한다.

```bash
ros2 topic pub --once /lane_state std_msgs/msg/Bool '{data: true}'
ros2 topic echo /lane_fusion/cmd_vel_preview
```

디버그 영상 토픽은 `/lane_fusion/debug/compressed`다. 기본은 비활성화이며 활성화 시 2Hz로 제한한다.
`/lane_mode`는 기존 연동용으로 받되 색 또는 영상 반쪽 선택에 사용하지 않는다. 값이 바뀌면 추적을 초기화한다.
`/intersection_state=true`는 새 입력부터 속도·각속도에 같은 비율을 곱한다.

실차에서 운영자가 저속 검증을 시작할 때만 기존 차선 제어기를 종료한 뒤 다음과 같이 구동 출력을 선택한다.
여기서 실행한 작업에는 실제 로봇 연결이나 바퀴 구동이 없다.

```bash
ros2 launch lane_detection_fusion fusion.launch.py drive_enabled:=true
# 초기 상태는 여전히 비활성. 주행 시작/중단은 /lane_state true/false
```

## 설정과 시간 조건

`config/profile.json`을 복사하고 `vision`에 바꿀 항목을 넣는다. 노드를 재시작하면 적용된다.
설정 예: `max_speed`, `max_angular`, `kp`, `k_heading`, `kd`, `roi_top`, `near_y`, `far_y`, `min_contrast`, `association_distance`, `max_width_age_s`.
생략한 값의 정확한 목록은 `core.FusionConfig`, 재생 결과의 `resolved_vision_config`에 있다.
전체 항목이 들어 있는 `config/profile.example.json`을 복사해 수정해도 된다. 별도 프로필은 `--ros-args -p profile:=/절대경로/profile.json`으로 선택한다.
기본 상한 0.10m/s, 0.8rad/s와 시간 임계값은 **초기 시험값이며 로봇 한계나 최적값을 확인한 수치가 아니다**.

수신 대기는 가장 최근 1장만 유지한다. 명령은 새 유효 결과가 끝나면 발행하며, `control_hz=20`은 독립 감시 주기다.
이전 명령을 반복 발행해 관측 시각을 새로 갱신하지 않는다. 수신 나이·원 header 나이·카메라 침묵 중 하나라도 제한을 넘으면 0을 발행한다.
중단/모드 변경/초기화 중 끝난 이전 계산은 세대 번호가 달라 폐기한다.

기본 `check_header_age=true`는 카메라와 노드의 ROS 시계가 일치해야 한다.
bag을 ROS로 재생한다면 별도 분석 환경에서 `/camera_raw_16_9`만 선택하고 `--clock` 및 노드 `use_sim_time=true`를 맞춘다.
bag의 기록된 `/cmd_vel`은 재생 대상에서 제외한다. 아래 오프라인 도구는 ROS 토픽을 전혀 발행하지 않는다.

## ROS 없이 BAG_RAW 재처리

필요한 Python 패키지는 numpy, opencv-python 또는 opencv-python-headless다. 영상 저장에는 imageio-ffmpeg가 추가로 필요하다.

```bash
python tools/check_local.py
python tools/replay_raw.py --bag-dir /path/to/bag_raw/normal --out ./replays/normal --video
python tools/replay_raw.py --bag-dir /path/to/bag_raw/reverse --out ./replays/reverse --video
```

`frames.csv`에 인지 상태·새 제어 미리보기·오류 이유·처리 시간을, `summary.json`에 설정·코드 해시·상태 수를 기록한다.
수동 조종으로 기록된 `/cmd_vel`이나 `/odom`을 새 코드의 명령·궤적으로 취급하지 않는다.
영상은 입력과 새 경계 추정의 비교이며, 실제 주행 시뮬레이션이 아니다.

## 확인한 범위와 다음 단계

자동 검사는 한 선 전환의 중심 연속성, 좌우 조향 부호, 소실 시 정지, 최신 입력 교체, 느린 결과 폐기, 중단 경합, 시계 점프를 확인한다.
단위 검사는 검출 정확도나 실제 주행 성공을 보증하지 않는다. 분석 PC 처리 시간도 로봇 전체 지연이 아니다.

이번 결과는 동작 가능한 개선 후보다. **bag 자동 파라미터 최적화, 실시간 카메라 노출 제어, YOLO 학습, 미터 기반 경로 제어는 아직 구현하지 않았다.**
현장에서는 반사·직선·양방향 커브 구간을 분리해 후보를 조정하고, 사용하지 않은 연속 구간을 확인한 뒤 낮은 속도에서 실차 검증한다.
색 정보가 포화돼 사라진 경우 기존 영상만으로 원래 색을 복원할 수 없다. 구조적인 반사가 계속 선처럼 검출되면 해당 실패 장면을 라벨링해 세그멘테이션 후보를 별도로 비교한다.

기존 원본과 다른 사용자의 저장소는 수정하지 않았다. 이 폴더는 새 로컬 Git 저장소이며 원격 조직 저장소 생성 상태는 회의 기록에 표시한다.
