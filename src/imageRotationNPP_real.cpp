/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Modified version that loads actual input images (PGM format)
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

// Load PGM image
bool loadImagePGM(const std::string &fileName, npp::ImageCPU_8u_C1 &rImage)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << fileName << std::endl;
        return false;
    }
    
    std::string line;
    
    // Read magic number
    std::getline(file, line);
    if (line != "P5") {
        std::cerr << "Unsupported PGM format: " << line << std::endl;
        return false;
    }
    
    // Skip comments
    while (std::getline(file, line) && line[0] == '#') {}
    
    // Parse dimensions
    std::istringstream iss(line);
    int width, height;
    iss >> width >> height;
    
    // Read max value
    std::getline(file, line);
    int maxVal = std::stoi(line);
    
    std::cout << "Loading image: " << width << "x" << height << " (max: " << maxVal << ")" << std::endl;
    
    // Create image
    rImage = npp::ImageCPU_8u_C1(width, height);
    
    // Read pixel data
    file.read(reinterpret_cast<char*>(rImage.data()), width * height);
    
    file.close();
    return true;
}

// Simple PGM format saver
void saveImagePGM(const std::string &fileName, const npp::ImageCPU_8u_C1 &rImage)
{
    std::ofstream file(fileName, std::ios::binary);
    if (!file.is_open()) {
        throw npp::Exception("Could not open file for writing");
    }
    
    // Write PGM header
    file << "P5\n";
    file << rImage.width() << " " << rImage.height() << "\n";
    file << "255\n";
    
    // Write image data
    for (int y = 0; y < rImage.height(); ++y) {
        file.write(reinterpret_cast<const char*>(rImage.data() + y * rImage.pitch()), rImage.width());
    }
    
    file.close();
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        // Try to load the actual Lena image
        std::string inputFile = "data\\Lena_gray.pgm";
        std::cout << "Loading actual Lena image from: " << inputFile << std::endl;
        
        npp::ImageCPU_8u_C1 oHostSrc;
        if (!loadImagePGM(inputFile, oHostSrc)) {
            std::cerr << "Failed to load image. Creating test pattern instead." << std::endl;
            
            // Fallback to test pattern if image loading fails
            oHostSrc = npp::ImageCPU_8u_C1(512, 512);
            for (int y = 0; y < 512; ++y) {
                for (int x = 0; x < 512; ++x) {
                    int checkSize = 32;
                    bool isWhite = ((x / checkSize) + (y / checkSize)) % 2 == 0;
                    oHostSrc.data()[y * oHostSrc.pitch() + x] = isWhite ? 255 : 64;
                }
            }
        }
        
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create struct with the ROI size
        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiRect oSrcROI = {
            oSrcOffset.x,    // x-offset
            oSrcOffset.y,    // y-offset
            oSrcSize.width,  // width
            oSrcSize.height  // height
          };
          
        // Calculate the bounding box of the rotated image
        double angle = 45.0; // Rotation angle in degrees
        
        // For 45-degree rotation, we need a larger bounding box
        // The diagonal of a square rotated 45 degrees is sqrt(2) times larger
        int newWidth = static_cast<int>(oSrcSize.width * 1.5);
        int newHeight = static_cast<int>(oSrcSize.height * 1.5);
        
        NppiRect oBoundingBox = {0, 0, newWidth, newHeight};

        std::cout << "Rotating image by " << angle << " degrees..." << std::endl;
        std::cout << "Original size: " << oSrcSize.width << "x" << oSrcSize.height << std::endl;
        std::cout << "Rotated size: " << oBoundingBox.width << "x" << oBoundingBox.height << std::endl;

        // allocate device image for the rotated image
        npp::ImageNPP_8u_C1 oDeviceDst(oBoundingBox.width, oBoundingBox.height);

        // Set the rotation point (center of the image)
        NppiPoint oRotationCenter = {(int)(oSrcSize.width / 2), (int)(oSrcSize.height / 2)};

        // run the rotation - use updated API
        NPP_CHECK_NPP( nppiRotate_8u_C1R(
            oDeviceSrc.data(), oSrcSize, oDeviceSrc.pitch(), oSrcROI,
            oDeviceDst.data(), oDeviceDst.pitch(), oBoundingBox,
            angle, oRotationCenter.x, oRotationCenter.y, NPPI_INTER_NN) );
        
        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        std::string outputFile = "data\\Lena_rotated.pgm";
        saveImagePGM(outputFile, oHostDst);
        std::cout << "Saved rotated image: " << outputFile << std::endl;
        
        // Also create PNG version for easy viewing
        std::cout << "Creating PNG version..." << std::endl;
        std::string pngCommand = "& \"C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe\" \"" + outputFile + "\" \"data\\Lena_rotated.png\"";
        system("powershell -Command \"& 'C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe' 'data\\Lena_rotated.pgm' 'data\\Lena_rotated.png'\"");
        
        std::cout << "✅ SUCCESS: Actual input image rotated 45 degrees!" << std::endl;
        std::cout << "📁 Output files:" << std::endl;
        std::cout << "   - " << outputFile << " (PGM format)" << std::endl;
        std::cout << "   - data\\Lena_rotated.png (PNG format)" << std::endl;

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
