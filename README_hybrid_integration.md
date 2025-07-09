# SuperGlue + COLMAP 하이브리드 파이프라인 통합 가이드

이 가이드는 **SuperGlue + COLMAP 하이브리드 파이프라인**을 **3DGS train.py**와 통합하여 사용하는 방법을 설명합니다.

## 🎯 통합 개요

```
입력 이미지 → SuperGlue 특징점 추출 → COLMAP SfM → 3DGS 학습 → 결과
```

## 🚀 빠른 시작

### 1. 기본 사용법
```bash
# 전체 파이프라인 실행 (SfM + 3DGS 학습)
python run_hybrid_pipeline.py --source_path ./ImageInputs/images

# SfM만 실행 (3DGS 학습 건너뛰기)
python run_hybrid_pipeline.py --source_path ./ImageInputs/images --skip_training
```

### 2. 고급 옵션
```bash
python run_hybrid_pipeline.py \
    --source_path ./ImageInputs/images \
    --model_path ./my_output \
    --max_images 150 \
    --superglue_config outdoor \
    --iterations 50000 \
    --device cuda
```

### 3. 직접 train.py 사용
```bash
# SuperGlue + COLMAP 하이브리드 파이프라인으로 학습
python train.py \
    --source_path ./ImageInputs/images \
    --model_path ./output \
    --scene_type SuperGlueCOLMAPHybrid \
    --max_images 100 \
    --iterations 30000
```

## 📋 매개변수 설명

### run_hybrid_pipeline.py 옵션
- `--source_path`: 입력 이미지 디렉토리 (필수)
- `--model_path`: 출력 모델 디렉토리 (기본값: 자동 생성)
- `--max_images`: 최대 처리 이미지 수 (기본값: 100)
- `--superglue_config`: SuperGlue 설정 (indoor/outdoor, 기본값: outdoor)
- `--iterations`: 3DGS 학습 반복 수 (기본값: 30000)
- `--device`: GPU 디바이스 (기본값: cuda)
- `--skip_training`: SfM만 실행하고 3DGS 학습 건너뛰기

### train.py 추가 옵션
- `--scene_type SuperGlueCOLMAPHybrid`: 하이브리드 파이프라인 사용
- `--superglue_config`: SuperGlue 설정
- `--max_images`: 최대 이미지 수

## 🔧 파이프라인 단계

### 1단계: SuperGlue + COLMAP 하이브리드 SfM
1. **이미지 수집**: 품질 기반 필터링
2. **SuperGlue 특징점 추출**: 고품질 특징점 생성
3. **SuperGlue 매칭**: 강력한 특징점 매칭
4. **COLMAP SfM**: 안정적인 카메라 포즈 추정
5. **언디스토션**: 렌즈 왜곡 보정
6. **3DGS 형식 변환**: SceneInfo 생성

### 2단계: 3DGS 학습
1. **Gaussian 모델 초기화**: 포인트 클라우드 기반
2. **카메라 로딩**: SfM 결과 사용
3. **학습 실행**: 3D Gaussian Splatting
4. **결과 저장**: 최종 모델

## 📁 출력 구조

```
output/
├── superglue_colmap_hybrid_output/  # SfM 결과
│   ├── input/                       # 전처리된 이미지
│   ├── database.db                  # COLMAP 데이터베이스
│   ├── sparse/                      # SfM 결과
│   └── undistorted/                 # 언디스토션된 이미지
├── point_cloud/                     # 3DGS 포인트 클라우드
├── cameras.json                     # 카메라 정보
└── [학습 결과들...]
```

## 🎨 최적화 팁

### 성능 최적화
```bash
# GPU 메모리 절약
--max_images 50 --device cuda

# 빠른 테스트
--max_images 20 --iterations 1000 --skip_training
```

### 품질 향상
```bash
# 더 많은 이미지 사용
--max_images 200

# 더 긴 학습
--iterations 50000

# 실내 설정
--superglue_config indoor
```

## 🔍 문제 해결

### COLMAP 오류
```bash
# COLMAP 설치 확인
colmap --help

# 권한 문제
sudo chmod +x /usr/local/bin/colmap
```

### 메모리 부족
```bash
# 이미지 수 줄이기
--max_images 30

# CPU 사용
--device cpu
```

### SuperGlue 오류
```bash
# 모델 다운로드 확인
ls Superglue/models/weights/

# 설정 변경
--superglue_config indoor
```

## 📊 성능 비교

| 방법 | 특징점 품질 | 매칭 정확도 | 처리 속도 | 안정성 | 3DGS 호환성 |
|------|-------------|-------------|-----------|--------|-------------|
| COLMAP만 | 보통 | 보통 | 빠름 | 높음 | 높음 |
| SuperGlue만 | 높음 | 높음 | 느림 | 보통 | 보통 |
| **하이브리드** | **높음** | **높음** | **보통** | **높음** | **높음** |

## 🎯 권장 사용 사례

### ✅ 적합한 경우
- 모션 블러가 있는 이미지
- 복잡한 장면 구조
- 고품질 3D 재구성 필요
- 안정적인 결과 요구
- 3DGS 학습 준비 완료 필요

### ❌ 부적합한 경우
- 매우 빠른 처리 필요
- 단순한 장면 구조
- 제한된 컴퓨팅 리소스

## 🔗 관련 파일

- `superglue_colmap_hybrid.py`: 하이브리드 파이프라인 구현
- `run_hybrid_pipeline.py`: 통합 실행 스크립트
- `scene/dataset_readers.py`: 데이터셋 로더 통합
- `arguments/__init__.py`: 명령행 옵션 추가

## 📝 예시 명령

### 기본 실행
```bash
python run_hybrid_pipeline.py --source_path ./ImageInputs/images
```

### 고품질 설정
```bash
python run_hybrid_pipeline.py \
    --source_path ./ImageInputs/images \
    --max_images 200 \
    --superglue_config outdoor \
    --iterations 50000
```

### 빠른 테스트
```bash
python run_hybrid_pipeline.py \
    --source_path ./ImageInputs/images \
    --max_images 20 \
    --iterations 1000 \
    --skip_training
```

### 직접 train.py 사용
```bash
python train.py \
    --source_path ./ImageInputs/images \
    --model_path ./output \
    --scene_type SuperGlueCOLMAPHybrid \
    --max_images 100 \
    --iterations 30000
```

## 🎉 성공 메시지

파이프라인이 성공적으로 완료되면 다음과 같은 메시지가 표시됩니다:

```
🎉 모든 파이프라인 성공!
📁 결과 위치: ./output/abc123def4
📊 다음 명령으로 결과 확인:
   python train.py --source_path ./ImageInputs/images --model_path ./output/abc123def4 --scene_type SuperGlueCOLMAPHybrid
``` 