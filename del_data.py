"""
'.'으로 시작하는 파일들을 제거
"""

# 라이브러리
import os
import shutil

# 디렉토리 경로
image_data_dir = '/Volumes/Hyun/data/image_data'
label_data_dir = '/Volumes/Hyun/data/label_data'
del_data_dir = '/Volumes/Hyun/data/del_data'

# 새 디렉토리가 없다면 생성
os.makedirs(del_data_dir, exist_ok=True)

# 디렉토리에 있는 파일 가져오기
image_files = os.listdir(image_data_dir)
label_files = os.listdir(label_data_dir)

# 파일 옮기는 함수
def move_file(source_dir, file_name, target_dir):
    source_path = os.path.join(source_dir, file_name)
    target_path = os.path.join(target_dir, file_name)
    shutil.move(source_path, target_path)

# 이미지 디렉토리의 파일 중 .으로 시작하는 파일을 새 디렉토리로 옮기기
for image_file in image_files:
    if image_file.startswith('.'):
        move_file(image_data_dir, image_file, del_data_dir)

# 라벨 디렉토리의 파일 중 .으로 시작하는 파일을 새 디렉토리로 옮기기
for label_file in label_files:
    if label_file.startswith('.'):
        move_file(label_data_dir, label_file, del_data_dir)

print("이동 완료")