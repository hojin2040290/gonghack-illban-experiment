import cv2
import numpy as np
import os

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 사진을 저장할 폴더 경로
    save_dir = 'right_captured_images'

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_count = 0  # 사진 번호를 위한 카운터

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 이미지를 읽을 수 없습니다.")
            break

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 이진화
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 팽창 (경계 확장)
        dilated = cv2.dilate(thresh, None, iterations=2)

        # 침식 (경계 축소)
        eroded = cv2.erode(dilated, None, iterations=2)

        # 결과 이미지 출력
        cv2.imshow('Dilate & Erode', eroded)

        # 's' 키를 눌러서 다중 사진 저장
        if cv2.waitKey(1) & 0xFF == ord('s'):
            image_count += 1  # 이미지 번호 증가
            filename = os.path.join(save_dir, f'captured_image_{image_count}.png')
            cv2.imwrite(filename, eroded)  # 필터가 적용된 이미지 저장
            print(f'사진이 저장되었습니다: {filename}')
        
        # 'q'를 눌러서 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
