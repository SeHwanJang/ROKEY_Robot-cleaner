import cv2
import numpy as np
from matplotlib import pyplot as plt

# 비디오 캡처
cap = cv2.VideoCapture(0)

# 첫 번째 프레임 읽기
ret, frame = cap.read()
if not ret:
    print("비디오 캡처 실패")
    cap.release()
    exit()

# 첫 번째 프레임을 그레이스케일로 변환
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 첫 번째 프레임에서 특징점 찾기
feature_params = dict(maxCorners=100, qualityLevel=0.5, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 모든 특징점에 동일한 색상 할당 (예: 빨간색)
if p0 is not None:
    point_colors = np.array([[0, 0, 255]] * len(p0))  # 빨간색 (BGR)

# 최대 특징점 개수 설정
MAX_FEATURES = 100
MIN_DISTANCE = 30  # 새 특징점이 기존 점과 너무 가까운 경우 제외할 거리

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임을 그레이스케일로 변환
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 옵티컬 플로우 계산
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)

        if p1 is None or status is None:
            print("옵티컬 플로우 계산 실패")
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if p0 is not None:
                point_colors = np.array([[0, 0, 255]] * len(p0))
            continue

        status = status.reshape(-1)
        good_new = p1[status == 1].reshape(-1, 2)
        good_old = p0[status == 1].reshape(-1, 2)

        # point_colors 크기를 good_new에 맞게 조정
        point_colors = np.array([[0, 0, 255]] * len(good_new))

        # 새로 들어온 특징점 찾기
        p0_new = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        new_points = []
        if p0_new is not None:
            p0_new = p0_new.reshape(-1, 2)
            for new_point in p0_new:
                if all(np.linalg.norm(new_point - old_point) >= MIN_DISTANCE for old_point in good_new):
                    new_points.append(new_point)

        if new_points:
            combined_points = np.vstack((good_new, np.array(new_points)))
            if len(combined_points) > MAX_FEATURES:
                p0 = combined_points[:MAX_FEATURES]
            else:
                p0 = combined_points
        else:
            p0 = good_new

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), point_colors[i].tolist(), 2)

        for i, (new) in enumerate(good_new):
            a, b = new.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 5, point_colors[i].tolist(), -1)

        if new_points:
            for i, new in enumerate(new_points):
                a, b = new
                frame = cv2.circle(frame, (int(a), int(b)), 5, point_colors[i].tolist(), -1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.pause(0.01)
        plt.clf()

        old_gray = frame_gray.copy()
        p0 = np.vstack((good_new, np.array(new_points))) if new_points else good_new

except KeyboardInterrupt:
    print("사용자가 프로그램을 종료했습니다.")
finally:
    cap.release()
    plt.close()

