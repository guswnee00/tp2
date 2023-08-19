"""
기존 라벨 데이터에서 바운딩박스와 관련된 
pm_code, shape_type, points에 대한 정보만 추출해 새로운 json 파일로 저장
"""

# 라이브러리 불러오기
import os
import json

#데이터 루트 설정
d_root = '/Volumes/Hyun/data'

# 이미지와 라벨 데이터 디렉토리
image_data_dir = os.path.join(d_root, 'image_data')
label_data_dir = os.path.join(d_root, 'label_data')

# 이미지/라벨 파일 목록 가져오기
image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.jpg')]
label_files = [f for f in os.listdir(label_data_dir) if f.endswith('.json')]

# 새로운 JSON 파일들을 저장할 디렉토리 경로
new_label_data_dir = os.path.join(d_root, 'new_label_data')

# 새 디렉토리 생성
os.makedirs(new_label_data_dir, exist_ok=True)

# pm_code 값 매핑
pm_code_mapping = {
    '13': 0, '14': 1, '15': 2, '16': 3, '17': 4,
    '18': 5, '19': 6, '20': 7, '21': 8, '22': 9,
    '23': 10, '24': 11, '25': 12, '26': 13, '27': 14,
    '28': 15, '29': 16, '30': 17, '31': 18, '32': 19,
    '33': 20, '35': 21, '36': 22
}

# 기존 JSON 파일들 목록 얻기
for label_file in label_files:
    # JSON 파일 읽기
    with open(os.path.join(label_data_dir, label_file), 'r') as file:
        json_data = json.load(file)

    # PM 관련 정보 추출
    pm_annotations = json_data['annotations'].get('PM', [])

    # PM 관련 정보가 있는 경우
    if pm_annotations:
        # 새로운 파일로 저장할 JSON 데이터 생성
        new_pm_annotations = []
        for pm_info in pm_annotations:
            pm_code = pm_info['PM_code']
            shape_type = pm_info['shape_type']
            points = pm_info['points']
            
            # pm_code에 해당하는 매핑 값 가져오기
            mapped_pm_code = pm_code_mapping.get(pm_code, 99)

            new_pm_info = {
                "pm_code": mapped_pm_code,
                "shape_type": shape_type,
                "points": points
            }

            new_pm_annotations.append(new_pm_info)
        
        new_json_data = {
            "annotations": {"PM": new_pm_annotations}
        }

        # 새로운 파일로 저장
        new_label_path = os.path.join(new_label_data_dir, label_file)
        with open(new_label_path, 'w') as new_label_file:
            json.dump(new_json_data, new_label_file, indent=4)

# 확인
new_label_files = [f for f in os.listdir(new_label_data_dir) if f.endswith('.json')]
print(len(new_label_files))
print(new_label_files[0])
print('생성 완료')