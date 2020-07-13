# CUDA/PyQt Path Tracer demo
![Converged](demo/converged.png)

A small program to demonstrate wrapping an interactive CUDA path tracer with a PyQt UI.

## Build instructions
```
git clone --recursive https://github.com/chellmuth/cuda-pyqt-demo.git
cd cuda-pyqt-demo
pip install -r requirements.txt
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
PYTHONPATH=./build python app.py
```

## Technical details
* Forward path tracer with sphere and triangle primitives, diffuse brdfs, and environment lighting
* CUDA buffer shared directly with OpenGL to avoid overhead displaying the rendered image  
* Scene interaction instrumented through Python via `pybind11` bindings

## Features
### BRDF Updates
![BRDF](demo/brdf.gif)

### Geometry Updates
![Geometry](demo/rotate.gif)
