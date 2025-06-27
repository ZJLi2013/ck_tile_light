# ck_tile_light

light version of [composable kernels](https://github.com/ROCm/composable_kernel), keep only ck_tile

# build and run 

* choose the basic rocm/pytorch image: `rocm/pytorch:latest` 

```sh
docker run --rm -it  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /home/amd/github:/workspace/ -w /workspace rocm/pytorch:latest 

git clone https://github.com/ZJLi2013/ck_tile_light.git
cd ck_tile_light && \
mkdir build && \
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc .. 
# Build Single Example
cmake --build . --target tile_example_gemm_basic
./example/ck_tile/03_gemm/tile_example_gemm_basic

# Build all examples once for all (hours need)
# cmake --build . --target all 

# Install headers 
cmake --install . 
``` 


# TODO 

* add sink-attn, sage-attn pipelines


# Existing Bugs

moe related examples has build errors