"""
이미지파일과 라벨링파일은 동일한 이름에 확장자만 다름
따라서 같은 이름끼리 매칭시키고 짝이 없는 파일은 del_data 디렉토리로 이동시킴
"""

# 라이브러리
import os
import shutil

# 디렉토리 경로
image_data_dir = '/Volumes/Hyun/data/image_data'
label_data_dir = '/Volumes/Hyun/data/label_data'
del_data_dir = '/Volumes/Hyun/data/del_data'

# 디렉토리에 있는 파일 가져오기
image_files = os.listdir(image_data_dir)
label_files = os.listdir(label_data_dir)

# 파일 이름에서 확장자 제거하고 이름만 추출하는 함수
def get_file_name(file_name):
    return os.path.splitext(file_name)[0]

# 이미지와 라벨 파일 이름을 매칭시키는 딕셔너리
matching_dict = {}
for image_file in image_files:
    file_name = get_file_name(image_file)
    matching_dict[file_name] = {"image": image_file, "label": None}

for label_file in label_files:
    file_name = get_file_name(label_file)
    if file_name in matching_dict:
        matching_dict[file_name]["label"] = label_file

# del_data 디렉토리가 없다면 생성
os.makedirs(del_data_dir, exist_ok = True)

# 매칭이 안되는 파일을 del_data 디렉토리로 옮기기
for file_name, files in matching_dict.items():
    if files["image"] is None or files["label"] is None:
        if files["image"]:
            shutil.move(os.path.join(image_data_dir, files["image"]), os.path.join(del_data_dir, files["image"]))
        if files["label"]:
            shutil.move(os.path.join(label_data_dir, files["label"]), os.path.join(del_data_dir, files["label"]))

print("이동 완료")