import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO


def find_largest_similar_region_in_bbox(image, bbox):
    """
    바운딩 박스 영역에서 동일한 색상을 가진 가장 큰 연결 영역을 찾는 함수

    Args:
        image: 원본 이미지
        bbox: 바운딩 박스 (x1, y1, x2, y2)

    Returns:
        mask_img: 원본 이미지 크기의 단일 채널 바이너리 마스크(255=인페인팅 영역, 0=유지할 영역)
    """

    if bbox is None:        
        height, width = image.shape[:2]

        # 이미지를 평평한 2D 배열로 재구성 (각 픽셀은 BGR 값을 가짐)
        pixels = image.reshape(-1, 3)

        # 고유한 색상 값을 찾음
        unique_colors = np.unique(pixels, axis=0)

        max_area = 0
        mask_img = np.zeros((height, width), dtype=np.uint8)

        # 각 고유 색상에 대해 영역 크기 계산
        for color in unique_colors:
            # 현재 색상에 대한 마스크 생성
            color_mask = np.all(image == color, axis=2).astype(np.uint8) * 255

            # 연결된 컴포넌트 찾기
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                color_mask
            )

            # 배경(0)을 제외한 가장 큰 컴포넌트 찾기
            if num_labels > 1:
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                area = stats[largest_label, cv2.CC_STAT_AREA]

                # 현재까지 찾은 가장 큰 영역보다 크면 업데이트
                if area > max_area:
                    max_area = area
                    mask_img = np.zeros((height, width), dtype=np.uint8)
                    mask_img[labels == largest_label] = 255

        return mask_img

    x1, y1, x2, y2 = bbox
    # 바운딩 박스 영역 자르기
    roi = image[int(y1) : int(y2), int(x1) : int(x2)]

    # 이미지에서 고유한 색상 값 찾기
    pixels = roi.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    max_area = 0
    roi_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)

    # 각 고유 색상에 대해 영역 크기 계산
    for color in unique_colors:
        # 현재 색상에 대한 마스크 생성
        color_mask = np.all(roi == color, axis=2).astype(np.uint8) * 255

        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            color_mask
        )

        # 배경(0)을 제외한 가장 큰 컴포넌트 찾기
        if num_labels > 1:
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            area = stats[largest_label, cv2.CC_STAT_AREA]

            # 현재까지 찾은 가장 큰 영역보다 크면 업데이트
            if area > max_area:
                max_area = area
                roi_mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                roi_mask[labels == largest_label] = 255

    # 최종 결과를 원본 이미지 크기의 단일 채널 마스크에 복사
    full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    full_mask[int(y1) : int(y2), int(x1) : int(x2)] = roi_mask

    return full_mask

def process_image(yolo_model, simple_lama, image_path, name):
    """
    단일 이미지를 처리하는 함수

    Args:
        yolo_model: YOLO 모델
        simple_lama: SimpleLama 모델
        image_path: 처리할 이미지 경로
        name: 이미지 파일 이름
    """
    try:
        results = yolo_model(image_path + "/" + name, verbose=False)

        image = cv2.imread(image_path + "/" + name)

        for result in results:
            # yolo 영역 찾기
            boxes = result.boxes
            xyxy = boxes.xyxy

            # 찾은 영역 중 마스킹 찾기
            mask_img = find_largest_similar_region_in_bbox(image, xyxy[0] if len(xyxy) > 0 else None)

            # 마스킹 NP로 변경
            mask_np = np.array(mask_img)

            # lama 실행
            img_np = np.array(Image.open(image_path + "/" + name).convert("RGB"))

            result = simple_lama(img_np, mask_np)
            result.save("./open/train_mask/" + name)

    except Exception as e:
        print(f"_______{name} 처리 중 오류 발생: {e}_______")
        return False
    
    return True

def worker_init():
    """Initialize the worker with the models"""
    global yolo_model, simple_lama
    yolo_model = YOLO("best.pt")
    simple_lama = SimpleLama()

def worker_process(args):
    """Worker function to process a single image"""
    img_path, img_name = args
    global yolo_model, simple_lama
    return process_image(yolo_model, simple_lama, img_path, img_name)

if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count
    
    # Make sure the output directory exists
    os.makedirs("./open/train_mask", exist_ok=True)
    
    img_path = "./open/train_input"
    mask_path = "./open/test_mask"

    img_names = os.listdir(img_path)
    count = len(img_names)
    
    # Determine the number of processes to use
    num_processes = min(cpu_count(), count)
    print(f"Using {num_processes} processes for image processing")
    
    # Create task arguments
    task_args = [(img_path, name) for name in img_names]
    
    # Create a multiprocessing pool with initialization
    with Pool(processes=num_processes, initializer=worker_init) as pool:
        # Process images in parallel with a progress bar
        results = list(tqdm(pool.imap(worker_process, task_args), total=count, desc="Processing images"))
