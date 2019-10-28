# sgm
Semi-Global Matching evaluated with kitti groundtruth.  sgm Algo is from gishi523.

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

Make sure your are in the `build` folder to run the executables.

### Running
```bash
./sgm [dir]

dir should be compatible with Kitti dataset.   
./sgm ../data/training/

result generated at results/ folder
