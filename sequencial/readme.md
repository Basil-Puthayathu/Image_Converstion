# Parallel Image Pipeline (pthreads + OpenCV)

Grayscale conversion + histogram equalization, parallelized with POSIX threads.

## Requirements

- CMake 3.10+
- A C++17 compiler (g++ or clang++)
- OpenCV (with dev headers)

Install OpenCV if you don't have it:

```bash
# Ubuntu/Debian
sudo apt install libopencv-dev cmake build-essential

# macOS
brew install opencv cmake
```

## Build

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This produces an executable called `main_parallel` inside the `build/` folder.

## Run

```bash
./main_sequential <input_image> <output_image>
```

- `input_image` — path to the image you want to process
- `output_image` — where to save the result (optional, defaults to `output_equalized_parallel.png`

### Example

Using the sample image included in `sample_input.png`:

```bash
./main_parallel ../test/sample.png ../output.png 
```

This reads `sample.png`, runs grayscale conversion + histogram equalization
using 4 threads, and writes the processed image to `results/output.png`.
The program also prints timing for each stage (load, grayscale, histogram
equalization, save) to the console.Any format of image expect raw image can be used.

## Output

All results should be saved into the `results/` folder, e.g.:

```
results/
├── sample_input.png      # original example image
└── sample_output.png     # equalized grayscale output
```
