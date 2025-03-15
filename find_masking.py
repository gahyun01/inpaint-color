import concurrent.futures
import glob
import multiprocessing
import os

import cv2
import numpy as np
from tqdm import tqdm


def find_largest_similar_region(image_path, grid_step=5):
    """
    이미지에서 동일한 색상을 가진 가장 큰 연결 영역을 찾는 함수

    Args:
        image_path: 이미지 파일 경로
        grid_step: 사용하지 않지만 기존 인터페이스 유지를 위해 남겨둠

    Returns:
        mask_img: 검은색 배경에 흰색으로 표시된 마스크 이미지
    """
    # 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    height, width = img.shape[:2]

    # 이미지를 평평한 2D 배열로 재구성 (각 픽셀은 BGR 값을 가짐)
    pixels = img.reshape(-1, 3)

    # 고유한 색상 값을 찾음
    unique_colors = np.unique(pixels, axis=0)

    max_area = 0
    mask_img = np.zeros((height, width), dtype=np.uint8)

    # 각 고유 색상에 대해 영역 크기 계산
    for color in unique_colors:
        # 현재 색상에 대한 마스크 생성
        color_mask = np.all(img == color, axis=2).astype(np.uint8) * 255

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


def process_image(image_path, output_dir):
    """
    단일 이미지를 처리하는 함수

    Args:
        image_path: 처리할 이미지 경로
        output_dir: 출력 폴더 경로
    """
    try:
        # 연결된 유사 색상 영역 찾기
        mask_img = find_largest_similar_region(image_path)

        # 입력 파일명 추출
        input_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{input_filename}.npy")

        # 마스크 이미지 저장
        np.save(output_path, mask_img)
        return True
    except Exception as e:
        print(f"{image_path} 처리 중 오류 발생: {e}")
        return False


def main():
    # 입력 폴더 경로 설정
    input_dir = "open/test_input"

    # 출력 폴더 생성
    output_dir = "open/test_mask"
    os.makedirs(output_dir, exist_ok=True)

    # 입력 폴더의 모든 이미지 파일 가져오기
    image_files = glob.glob(os.path.join(input_dir, "*.png"))

    print(f"총 {len(image_files)}개 이미지 처리 시작...")

    # 사용할 CPU 코어 수 결정 (전체 코어의 80%를 사용)
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"병렬 처리 사용: {max_workers}개 코어")

    # 병렬 처리로 이미지 처리
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # 각 이미지에 대한 작업 제출
        futures = [
            executor.submit(process_image, img_path, output_dir)
            for img_path in image_files
        ]

        # tqdm을 사용하여 진행 상황 표시
        for _ in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            pass

    print("모든 이미지 처리 완료!")


if __name__ == "__main__":
    main()
