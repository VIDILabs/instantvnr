# Instant Neural Representation for Interactive Volume Rendering

### Qi Wu, David Bauer, Michael J. Doyle, and Kwan-Liu Ma

[Project Page](https://wilsoncernwq.github.io/publication/arxiv-instant-vnr), 
[ArXiv](https://arxiv.org/abs/2207.11620)

### Abstract

Neural networks have shown great potential in compressing volume data for visualization. However, due to the high cost of training and inference, such volumetric neural representations have thus far only been applied to offline data processing and non-interactive rendering. In this paper, we demonstrate that by simultaneously leveraging modern GPU tensor cores, a native CUDA neural network framework, and a well-designed rendering algorithm with macro-cell acceleration, we can interactively ray trace volumetric neural representations (10-60fps). Our neural representations are also high-fidelity (PSNR > 30dB) and compact (10-1000x smaller). Additionally, we show that it is possible to fit the entire training step inside a rendering loop and skip the pre-training process completely. To support extreme-scale volume data, we also develop an efficient out-of-core training strategy, which allows our volumetric neural representation training to potentially scale up to terascale using only an NVIDIA RTX 3090 workstation.


### Build Instructions

This project is expected to be built with our lightweight scientific visualization development framework: OVR. 

```
# Download the development framework
git clone --recursive https://github.com/wilsonCernWq/open-volume-renderer.git
cd open-volume-renderer/projects

# Download the source code
git clone --recursive https://github.com/wilsonCernWq/instant-vnr-engine.git
cd ..

# Build
mkdir build
cd build
cmake .. -DGDT_CUDA_ARCHITECTURES=86 -DOVR_BUILD_MODULE_NNVOLUME=ON -DOVR_BUILD_DEVICE_OSPRAY=OFF -DOVR_BUILD_DEVICE_OPTIX7=OFF
cmake --build . --config Release --parallel 16
```

### Execution
```

```

### Citation
```bibtex
@article{wu2022instant,
    title={Instant Neural Representation for Interactive Volume Rendering},
    author={Wu, Qi and Doyle, Michael J and Bauer, David and Ma, Kwan-Liu},
    journal={arXiv preprint arXiv:2207.11620},
    year={2022}
}
```