import cv2  # Install opencv-python
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model_path = "/Users/kimhojin/Documents/GitHub/gonghack-illban-experiment/models/keras_model.h5"
try:
    model = load_model(model_path, compile=False)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load model from {model_path}. Error: {e}")
    exit(1)

# 라벨 파일 로드
labels_path = "/Users/kimhojin/Documents/GitHub/gonghack-illban-experiment/labels.txt"
try:
    with open(labels_path, "r") as file:
        class_names = [line.strip() for line in file.readlines()]
    print(f"Labels loaded successfully from {labels_path}")
except FileNotFoundError:
    print(f"Labels file not found at {labels_path}")
    exit(1)

# 웹캠 초기화 (기본 카메라 사용)
camera = cv2.VideoCapture(0)  # 기본 카메라를 사용하려면 0을 입력
if not camera.isOpened():
    print("Failed to access the webcam.")
    camera.release()
    cv2.destroyAllWindows()
    exit(1)

try:
    while True:
        # 웹캠에서 이미지 읽기
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture an image from the webcam.")
            break

        # 이미지를 (224x224) 크기로 조정 (모델의 입력 크기에 맞춤)
        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

        # 필터 적용 (그레이스케일 -> 이진화 -> 팽창 -> 침식)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(thresh_frame, None, iterations=2)
        eroded_frame = cv2.erode(dilated_frame, None, iterations=2)

        # 필터 적용 이미지를 모델에 입력하기 위해 다시 3채널로 변환
        filtered_frame = cv2.cvtColor(eroded_frame, cv2.COLOR_GRAY2BGR)

        # 모델 입력 데이터 준비
        input_image = filtered_frame.astype(np.float32)
        input_image = np.expand_dims(input_image, axis=0)  # (1, 224, 224, 3)
        input_image = input_image / 255.0  # [0, 1] 범위로 정규화

        # 모델 예측
        prediction = model.predict(input_image, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # 결과 출력
        print(f"Class: {class_name}")
        print(f"Confidence Score: {confidence_score:.2%}")

        # 행동 결정
        if "전진" in class_name.lower():
            print("Moving Forward!")
        elif "좌회전" in class_name.lower():
            print("Turning Left!")
        elif "우회전" in class_name.lower():
            print("Turning Right!")
        else:
            print("Unknown action.")

        # 결과 이미지 표시
        cv2.imshow("Filtered Image", filtered_frame)

except KeyboardInterrupt:
    print("Interrupted manually by the user.")

finally:
    # 리소스 해제
    camera.release()
    cv2.destroyAllWindows()
    print("Camera and window resources released successfully.")
