# CUDA NPP Image Rotation Project

## Project Description

This project implements GPU-accelerated image rotation using NVIDIA Performance Primitives (NPP) library and CUDA. The program performs geometric transformation of grayscale images, specifically rotating input images by 45 degrees using GPU parallel processing capabilities.

## What This Project Does

1. **Loads a grayscale image** from PGM format or creates a test pattern
2. **Uploads image data** from CPU memory to GPU memory
3. **Performs 45-degree rotation** using NPP's optimized GPU kernels
4. **Downloads the result** back to CPU memory
5. **Saves the rotated image** in PGM format
6. **Automatically converts** output to PNG for easy viewing

The rotation uses nearest-neighbor interpolation and calculates the appropriate bounding box to contain the entire rotated image.

## Project Structure

```
├── bin/                    # Compiled executables
│   ├── imageRotationNPP.exe           # Simple test version
│   └── imageRotationNPP_real.exe      # Full version with image loading
├── data/                   # Input and output images
│   ├── Lena.png           # Original color test image
│   ├── Lena_gray.pgm      # Grayscale version for processing
│   ├── Lena_rotated.pgm   # Output: rotated image (PGM format)
│   └── Lena_rotated.png   # Output: rotated image (PNG format)
├── src/                    # Source code
│   ├── imageRotationNPP.cpp           # Original NVIDIA template
│   ├── imageRotationNPP_simple.cpp    # Simplified test version
│   └── imageRotationNPP_real.cpp      # Working image loader version
├── common/include/         # NPP wrapper headers
├── Build.ps1              # Windows build script
├── Run.ps1               # Windows run script
├── ConvertToPNG.ps1      # PGM to PNG converter
└── Verify-VSSetup.ps1    # Visual Studio verification
```

## Key Learning Concepts

- **GPU Memory Management**: Host-to-device and device-to-host data transfers
- **NPP Library Usage**: Leveraging NVIDIA's optimized image processing primitives  
- **CUDA Integration**: Combining CUDA runtime with specialized libraries
- **Image Processing**: Geometric transformations and coordinate mapping
- **Performance Optimization**: GPU acceleration vs CPU processing

## CUDA APIs Used

- **Runtime API**: Device discovery, memory management, error handling
- **NPP Library**: `nppiRotate_8u_C1R` for optimized image rotation
- **Helper Functions**: CUDA device detection and NPP error checking

## Supported Hardware

- **GPU**: NVIDIA GPUs with compute capability 3.5 or higher
- **Architectures**: SM 3.5, 3.7, 5.0, 5.2, 6.0, 6.1, 7.0, 7.2, 7.5, 8.0, 8.6+
- **Operating Systems**: Windows 10/11, Linux x86_64

## Quick Start Guide

### Step 1: Check Prerequisites
1. **Verify you have an NVIDIA GPU**
   ```powershell
   nvidia-smi
   ```
   Should show your GPU information

2. **Check CUDA installation**
   ```powershell
   nvcc --version
   ```
   Should show CUDA compiler version

### Step 2: Download and Setup
1. **Clone or download this project**
2. **Navigate to project directory**
   ```powershell
   cd CUDAatScaleForTheEnterpriseCourseProjectTemplate
   ```

### Step 3: Install Dependencies (First-time setup)
1. **Install Visual Studio Community 2022**
   ```powershell
   winget install Microsoft.VisualStudio.2022.Community
   ```
   
2. **In Visual Studio Installer, add C++ workload:**
   - Open Visual Studio Installer
   - Click "Modify" next to Visual Studio Community 2022
   - Check "Desktop development with C++"
   - Click "Modify" to install

3. **Install ImageMagick (for PNG conversion)**
   ```powershell
   winget install ImageMagick.ImageMagick
   ```

### Step 4: Verify Installation
```powershell
powershell -ExecutionPolicy Bypass -File .\Verify-VSSetup.ps1
```
Should show: "Visual Studio C++ workload is properly installed!"

### Step 5: Prepare Input Image
```powershell
# Convert the provided Lena.png to grayscale PGM format
& "C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe" "data\Lena.png" -colorspace Gray "data\Lena_gray.pgm"
```

### Step 6: Build the Project
```powershell
powershell -ExecutionPolicy Bypass -File .\Build.ps1
```
Should show: "Build successful! Executable created: bin\imageRotationNPP.exe"

### Step 7: Run Image Rotation
```powershell
.\bin\imageRotationNPP_real.exe
```

### Step 8: View Results
- **PGM format**: `data\Lena_rotated.pgm`
- **PNG format**: `data\Lena_rotated.png` (automatically created)

You can open the PNG file with any image viewer to see the 45-degree rotated Lena image.

## System Requirements

### Hardware
- NVIDIA GPU with compute capability 3.5 or higher
- Compatible NVIDIA drivers

### Software
- Windows 10/11 or Linux
- CUDA Toolkit 11.4 or newer (includes NPP library)
- Visual Studio 2022 with C++ workload (Windows)
- GCC compiler (Linux)

### Windows Setup Instructions

1. **Install CUDA Toolkit**
   - Download from NVIDIA Developer website
   - Ensure NPP components are included

2. **Install Visual Studio Community 2022**
   ```powershell
   winget install Microsoft.VisualStudio.2022.Community
   ```
   - Select "Desktop development with C++" workload during installation

3. **Verify Installation**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\Verify-VSSetup.ps1
   ```


## Building and Running (Windows)

### Build the Project
```powershell
powershell -ExecutionPolicy Bypass -File .\Build.ps1
```

### Run the Program
```powershell
.\bin\imageRotationNPP_real.exe
```

### Alternative: Use Run Script
```powershell
powershell -ExecutionPolicy Bypass -File .\Run.ps1
```

## Input and Output

**Input**: The program loads `data/Lena_gray.pgm` (grayscale version of the famous Lena test image)
- Original dimensions: 1666x1250 pixels
- Format: PGM (Portable Graymap)

**Output**: Two files are generated:
- `data/Lena_rotated.pgm` - Raw PGM format
- `data/Lena_rotated.png` - PNG format for easy viewing
- Rotated dimensions: 2499x1875 pixels (larger to contain the rotated image)

## Technical Details

- **Rotation Angle**: Fixed at 45 degrees
- **Interpolation**: Nearest-neighbor (NPPI_INTER_NN)
- **Memory Management**: Automatic GPU memory allocation via NPP wrappers
- **Image Format**: 8-bit grayscale (single channel)
- **GPU Function Used**: `nppiRotate_8u_C1R`

## Example Output

```
D:\GPU_Trainee\CUDAatScaleForTheEnterpriseCourseProjectTemplate\bin\imageRotationNPP_real.exe Starting...

GPU Device 0: "Pascal" with compute capability 6.1

NPP Library Version 12.4.1
  CUDA Driver  Version: 12.9
  CUDA Runtime Version: 12.9
  Device 0: <          Pascal >, Compute SM 6.1 detected
Loading actual Lena image from: data\Lena_gray.pgm
Loading image: 1666x1250 (max: 255)
Rotating image by 45 degrees...
Original size: 1666x1250
Rotated size: 2499x1875
Saved rotated image: data\Lena_rotated.pgm
Creating PNG version...
SUCCESS: Actual input image rotated 45 degrees!
Output files:
   - data\Lena_rotated.pgm (PGM format)
   - data\Lena_rotated.png (PNG format)
```

## Troubleshooting

**Build fails with "cl.exe not found"**
- Run `.\Verify-VSSetup.ps1` to check Visual Studio installation
- Ensure "Desktop development with C++" workload is installed

**CUDA not found**
- Verify CUDA Toolkit installation
- Check `$env:CUDA_PATH` environment variable

**No GPU detected**
- Ensure NVIDIA drivers are installed
- Check GPU compute capability (must be 3.5+)

## Usage Scenarios

### Scenario 1: First Time User (Complete Setup)
```powershell
# 1. Check if you have NVIDIA GPU
nvidia-smi

# 2. Install dependencies
winget install Microsoft.VisualStudio.2022.Community
winget install ImageMagick.ImageMagick

# 3. Setup C++ workload in Visual Studio Installer
# 4. Verify setup
powershell -ExecutionPolicy Bypass -File .\Verify-VSSetup.ps1

# 5. Prepare input image
& "C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe" "data\Lena.png" -colorspace Gray "data\Lena_gray.pgm"

# 6. Build and run
powershell -ExecutionPolicy Bypass -File .\Build.ps1
.\bin\imageRotationNPP_real.exe
```

### Scenario 2: Quick Run (Dependencies Already Installed)
```powershell
# Just build and run
powershell -ExecutionPolicy Bypass -File .\Build.ps1
.\bin\imageRotationNPP_real.exe
```

### Scenario 3: Using Your Own Image
```powershell
# 1. Convert your image to grayscale PGM
& "C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe" "path\to\your\image.jpg" -colorspace Gray "data\Lena_gray.pgm"

# 2. Run the rotation
.\bin\imageRotationNPP_real.exe

# Your rotated image will be in data\Lena_rotated.png
```

### Scenario 4: Development/Testing
```powershell
# Clean previous builds
Remove-Item bin\*.exe -Force

# Build with specific compiler flags
powershell -ExecutionPolicy Bypass -File .\Build.ps1

# Run and check output
.\bin\imageRotationNPP_real.exe
dir data\*rotated*
```

## Alternative: Simple Test Version
If you want to see GPU rotation without image loading:
```powershell
.\bin\imageRotationNPP.exe
```
This creates a test checkerboard pattern and rotates it.

## What You Should See

1. **During execution**: GPU information, image dimensions, processing messages
2. **Output files**: 
   - `data\Lena_rotated.pgm` (raw format, ~2.3MB)
   - `data\Lena_rotated.png` (viewable format, ~1.8MB)
3. **Visual result**: The famous Lena image rotated 45 degrees clockwise
4. **Size change**: Original 1666×1250 becomes 2499×1875 (larger to fit rotation)

## Understanding the Output

### Performance Information
The program displays detailed information about your system:
```
GPU Device 0: "Pascal" with compute capability 6.1
NPP Library Version 12.4.1
CUDA Driver Version: 12.9
CUDA Runtime Version: 12.9
```

### Image Processing Details
```
Loading image: 1666x1250 (max: 255)
Rotating image by 45 degrees...
Original size: 1666x1250
Rotated size: 2499x1875
```

### File Outputs Explained
- **Original**: `data\Lena.png` (4.5MB color image)
- **Intermediate**: `data\Lena_gray.pgm` (2.1MB grayscale for processing)
- **Result PGM**: `data\Lena_rotated.pgm` (4.7MB raw rotated data)
- **Result PNG**: `data\Lena_rotated.png` (1.8MB compressed viewable image)

## Next Steps and Learning

### Experiment with the Code
1. **Change rotation angle**: Modify the `angle` variable in `imageRotationNPP_real.cpp`
2. **Try different interpolation**: Change `NPPI_INTER_NN` to `NPPI_INTER_LINEAR`
3. **Process your own images**: Replace Lena with your own grayscale images

### Understanding GPU Acceleration
- **CPU vs GPU**: Try processing large images to see performance difference
- **Memory transfers**: Notice the host-to-device and device-to-host operations
- **Parallel processing**: Each pixel rotation is computed in parallel on GPU cores

### Extend the Project
- Add support for color images using `nppiRotate_8u_C3R`
- Implement different rotation angles as command-line parameters
- Add batch processing for multiple images
- Measure and compare processing times

## Common Questions

**Q: Why convert to grayscale?**
A: This educational template focuses on the rotation algorithm, not color processing. NPP supports color rotation with `nppiRotate_8u_C3R`.

**Q: Can I rotate by other angles?**
A: Yes, modify the `angle` variable. The bounding box calculation may need adjustment for optimal results.

**Q: Why is the output larger?**
A: When rotating a rectangle, the diagonal becomes longer. The output size ensures no image data is lost.
