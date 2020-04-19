#include<opencv2/opencv.hpp>
#include<iostream>

__global__ void convert2gray(uchar3 *color_pixel, unsigned char* gray_pixel){
    int ID = blockIdx.x*blockDim.x+threadIdx.x;

    gray_pixel[ID] = (unsigned char)(0.299f*color_pixel[ID].x
            + 0.586f*(float)color_pixel[ID].y
            + 0.114f*(float)color_pixel[ID].z);
}

int main(){
    // nvcc grayscale.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
    cv::Mat input_img = cv::imread("dear.jpg",1);
    if(input_img.empty()==true){
        return 1;
    }

    int width = input_img.cols;
    int height = input_img.rows;
    std::cout << "Image_size : " << width << "×" << height << std::endl;

    uchar3* host_img_array_color = new uchar3[width*height];
    unsigned char* host_img_array_gray = new unsigned char[width*height];

    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            host_img_array_color[x+y*width]
            = make_uchar3(input_img.at<cv::Vec3b>(y, x)[2], input_img.at<cv::Vec3b>(y, x)[1], input_img.at<cv::Vec3b>(y, x)[0]);
        }
    }

    uchar3* device_img_array_color;
    unsigned char* device_img_array_gray;
    int datasize_color = sizeof(uchar3) * width * height;
    int datasize_gray = sizeof(unsigned char) * width * height;
    cudaMalloc((void**)&device_img_array_color, datasize_color);
    cudaMalloc((void**)&device_img_array_gray, datasize_gray);

    cudaMemcpy(device_img_array_color, host_img_array_color, datasize_color, cudaMemcpyHostToDevice);

    convert2gray<<<width*height,1>>> (device_img_array_color, device_img_array_gray);

    cudaMemcpy(host_img_array_gray, device_img_array_gray, datasize_gray, cudaMemcpyDeviceToHost);

    cv::Mat1b output_img(height, width);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output_img.at<unsigned char>(y, x) = host_img_array_gray[x + y * width];
        }
    }
    cv::imwrite("gray.jpg", output_img);

    cudaFree(device_img_array_color);
    cudaFree(device_img_array_gray);
    delete host_img_array_color;
    delete host_img_array_gray;

    return 0;
}