# HLOC + 3DGS Pipeline

## How to use

0. Go to the project root directory, and run **conda env create --file environment.yml** to create a virtual environment for this model.
1. Locate your desired images to ./ImageInputs/images/, and create a directory ./ImageInputs/sparse/0
2. In your project root directory, run **python hloc_pipeline.py**
    * This will generate ./ImageInputs/hloc_outputs
    * under this directory, you will find camera.bin, image.bin, and point.bin. move these files to ./ImageInputs/sparse/0
3. In your project root directory, run **python train.py -s ImageInputs/   --iterations (n)   -m output/(output_folder_name) --eval**
4. After done training, run **python render.py -m output/(output_folder_name)  --iteration (n)**
5. After rendering, run **python metrics.py -m output/(output_folder_name)**
