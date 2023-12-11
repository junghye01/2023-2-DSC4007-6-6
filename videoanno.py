import csv
import cv2
import os

# MP4 파일이 위치한 root 디렉토리
root_dir = "/home/irteam/junghye-dcloud-dir/dscapstone/KR-Anomaly-Detection-Dataset"

# 저장할 디렉토리
output_dir = "./video2frames_KR/train"

# 파일 리스트가 저장된 TXT 파일의 경로
file_list_txt = "/home/irteam/junghye-dcloud-dir/dscapstone/KR-Anomaly-Detection-Dataset/Annotation_Train.txt"

# 결과를 저장할 CSV 파일 경로
output_csv = "./video2frames_KR/train_anno.csv"

# CSV 파일 초기화
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["video_name", "paths","label"])

# TXT 파일에서 파일 리스트를 불러옴
with open(file_list_txt, 'r') as f:
    file_list = [line.strip() for line in f.readlines() if line.strip()]


# 각 파일에 대하여
for file_name in file_list:
    # 전체 디렉토리에서 파일을 검색
    file_name = file_name.split('/')[-1]
    for root, dirs, files in os.walk(root_dir):
        if file_name in files:
            full_path = os.path.join(root, file_name)
            # 비디오 파일을 읽음
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"Failed to open {full_path}")
                continue
            
            # 이하 생략...
            base_name = os.path.splitext(file_name)[0]
            dir_name = os.path.basename(os.path.dirname(full_path))
            # normal일 경우
            
            output_subdir = os.path.join(output_dir, dir_name, base_name)
            
            #print(base_name,dir_name,output_subdir)
            #CSV 파일에 기록
            frame_paths = []
            for frame_file in sorted(os.listdir(output_subdir)):
                if frame_file.endswith('.jpg'):
                    frame_paths.append(os.path.join(output_subdir, frame_file))

            with open(output_csv, 'a', newline='') as file:
                writer = csv.writer(file)
                for path in frame_paths:
                    if dir_name=='Training_Normal_Videos_Anomaly':
                        dir_name='Normal'
                    writer.writerow([file_name, path, dir_name])

            print(f"{full_path} has been processed and annotations saved to CSV.")
            break
    else:
        print(f"{file_name} not found in {root_dir}")

print("CSV creation complete!")
