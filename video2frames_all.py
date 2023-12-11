import cv2
import os

# MP4 파일이 위치한 root 디렉토리
root_dir = "/home/irteam/junghye-dcloud-dir/dscapstone/KR-Anomaly-Detection-Dataset"

# 저장할 디렉토리
output_dir = "./video2frames_KR/train"

# 파일 리스트가 저장된 TXT 파일의 경로
file_list_txt = "/home/irteam/junghye-dcloud-dir/dscapstone/KR-Anomaly-Detection-Dataset/Annotation_Train.txt"

# TXT 파일에서 파일 리스트를 불러옴
with open(file_list_txt, 'r') as f:
    file_list = [line.strip() for line in f.readlines() if line.strip()]

# 각 파일에 대하여
for file_name in file_list:
    # 전체 디렉토리에서 파일을 검색
    file_name=file_name.split('/')[-1]
    for root, dirs, files in os.walk(root_dir):
        if file_name in files:
            full_path = os.path.join(root, file_name)
            
            # 비디오 파일을 읽음
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"Failed to open {full_path}")
                continue
            
            # 비디오의 fps를 확인
           # fps=cap.get(cv2.CAP_PROP_FPS)
           # frame_interval=round(fps/16)
          #  if frame_interval<1:
          #      frame_interval=1
            
            # 출력 디렉토리 생성 (예: video2frames/test/Arson/Arson011_x264)
            base_name = os.path.splitext(file_name)[0] # 파일이름에서 확장자 분리 
            dir_name = os.path.basename(os.path.dirname(full_path))
            output_subdir = os.path.join(output_dir, dir_name, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 프레임 단위로 자르기
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                #if frame_num%frame_interval==0:
                # JPG 파일로 저장 (예: frame_00001.jpg)
                output_frame_path = os.path.join(output_subdir, f"frame_{frame_num:05d}.jpg")
                cv2.imwrite(output_frame_path, frame)
                
                frame_num += 1

            # 비디오 캡처 객체를 해제
            cap.release()
            print(f"{full_path} has been processed, {frame_num} frames were saved to {output_subdir}")
            break
    else:
        print(f"{file_name} not found in {root_dir}")

print("Processing complete!")
