# Interactive Volume Visualization Via Multi-Resolution Hash Encoding Based Neural Representation

### Qi Wu, David Bauer, Michael J. Doyle, and Kwan-Liu Ma

Published in: IEEE Transactions on Visualization and Computer Graphics ( Early Access )

[Project Page](https://wilsoncernwq.github.io/publication/tvcg2022-instant-vnr), 
[Github Page](https://github.com/VIDILabs/instantvnr), 
[ArXiv](https://arxiv.org/abs/2207.11620),
[Publishers' Version](https://ieeexplore.ieee.org/document/10175377)

### TODO

- [X] Complete documentations
- [X] Provide a docker container
- [ ] Document data configuration files
- [ ] Support isosurface
- [ ] Support rendering with isosurfaces
- [ ] Support rendering with unstructured data clipping
- [ ] More modular sampling and training API

### Abstract

Neural networks have shown great potential in compressing volume data for visualization. However, due to the high cost of training and inference, such volumetric neural representations have thus far only been applied to offline data processing and non-interactive rendering. In this paper, we demonstrate that by simultaneously leveraging modern GPU tensor cores, a native CUDA neural network framework, and a well-designed rendering algorithm with macro-cell acceleration, we can interactively ray trace volumetric neural representations (10-60fps). Our neural representations are also high-fidelity (PSNR > 30dB) and compact (10-1000x smaller). Additionally, we show that it is possible to fit the entire training step inside a rendering loop and skip the pre-training process completely. To support extreme-scale volume data, we also develop an efficient out-of-core training strategy, which allows our volumetric neural representation training to potentially scale up to terascale using only an NVIDIA RTX 3090 workstation.


### Build Instructions

This project is expected to be built with our lightweight scientific visualization development framework, [OVR](https://github.com/VIDILabs/open-volume-renderer). Our project currently require an NVIDIA GPU with compute compatibility >= 7.0.

#### Command Line

```
# Download the development framework
git clone --recursive https://github.com/VIDILabs/open-volume-renderer.git
cd open-volume-renderer/projects

# Download the source code
git clone --recursive https://github.com/VIDILabs/instantvnr.git
cd ..

# Build
mkdir build
cd build
cmake .. -DGDT_CUDA_ARCHITECTURES=86 -DOVR_BUILD_MODULE_NNVOLUME=ON -DOVR_BUILD_DEVICE_OSPRAY=OFF -DOVR_BUILD_DEVICE_OPTIX7=OFF
cmake --build . --config Release --parallel 16

# In the binary output directory, setup symbolic links to the data folder
ln -s ../../data .
cp ../../projects/instantvnr/example-model.json .

# Run Apps
./vnr_int_dual   --volume ./data/configs/scene_vorts1.json --network ./example-model.json --rendering-mode 5
./vnr_int_single --neural-volume ./params.json --tfn ./data/configs/scene_vorts1.json --rendering-mode 5
./vnr_cmd_train  --volume ./data/configs/scene_vorts1.json --network ./example-model.json 
./vnr_cmd_render --neural-volume ./params.json --tfn ./data/configs/scene_vorts1.json --rendering-mode 5 --num-frames 1
```

#### Docker Containers

We also provide a docker container to build the project and run our applications
```
# Build the docker container
git clone --recursive https://github.com/VIDILabs/instantvnr.git
cd instantvnr
docker build -t instantvnr --build-arg="CUDA_ARCH=86" .
xhost +si:localuser:root

# Launch an interactive docker session
docker run --gpus device=0 --runtime=nvidia -ti  --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -w /instantvnr/build instantvnr
```

You can also directly execute apps through the docker container


### Citation
```bibtex
@article{wu2022instant,
  author={Wu, Qi and Bauer, David and Doyle, Michael J. and Ma, Kwan-Liu},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={Interactive Volume Visualization Via Multi-Resolution Hash Encoding Based Neural Representation}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TVCG.2023.3293121}
}
```
