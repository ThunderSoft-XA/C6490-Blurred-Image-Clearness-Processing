#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>

#include "SNPETask.h"

std::unique_ptr<snpetask::SNPETask> m_task;
std::vector<std::string> inputLayers;
std::vector<std::string> outputLayers;
std::vector<std::string> outputTensors;

static void PreProcess(const cv::Mat& image)
{
    int inputHeight = 452;
    int inputWidth = 800;
    cv::Mat input0(inputHeight, inputWidth, CV_8UC3);
    cv::resize(image, input0, cv::Size(inputWidth, inputHeight), cv::INTER_LINEAR);
    float* input00 =  m_task->getInputTensor(inputLayers[0]);
    for (int i=0; i<800; i++) {
        for (int j=0; j<452; j++) {
            for (int k=0; k<3; k++) {
                *(input00 + i*452*3 + j*3 + k) = *(input0.data + i*452*3 + j*3 + k);
            }
        }
    }

    inputHeight = 226;
    inputWidth = 400;
    cv::Mat input1(inputHeight, inputWidth, CV_8UC3);
    cv::resize(image, input1, cv::Size(inputWidth, inputHeight), cv::INTER_LINEAR);
    float* input11 =  m_task->getInputTensor(inputLayers[1]);
    for (int i=0; i<400; i++) {
        for (int j=0; j<226; j++) {
            for (int k=0; k<3; k++) {
                *(input11 + i*226*3 + j*3 + k) = *(input1.data + i*226*3 + j*3 + k);
            }
        }
    }

    inputHeight = 113;
    inputWidth = 200;
    cv::Mat input2(inputHeight, inputWidth, CV_8UC3);
    cv::resize(image, input2, cv::Size(inputWidth, inputHeight), cv::INTER_LINEAR);
    float* input22 = m_task->getInputTensor(inputLayers[2]);
    for (int i=0; i<200; i++) {
        for (int j=0; j<113; j++) {
            for (int k=0; k<3; k++) {
                *(input22 + i*113*3 + j*3 + k) = *(input2.data + i*113*3 + j*3 + k);
            }
        }
    }
}


static void PostProcess()
{
    const float* predOutput0 = m_task->getOutputTensor(outputTensors[0]);
    const float* predOutput1 = m_task->getOutputTensor(outputTensors[1]);
    const float* predOutput2 = m_task->getOutputTensor(outputTensors[2]);

    char* p0 = (char*)malloc(452*800*3*sizeof(char));
    for (int i=0; i<452*800*3; i++) {
        int res = (int)(*(predOutput0+i) + 128);
        if (res > 255)
            res = 255;
        if (res < 0)
            res = 0;
        *(p0+i) = res;
    }

    char* p1 = (char*)malloc(226*400*3*sizeof(char));
    for (int i=0; i<226*400*3; i++) {
        int res = (int)(*(predOutput1+i) + 128);
        if (res > 255)
            res = 255;
        if (res < 0)
            res = 0;
        *(p1+i) = res;
    }

    char* p2 = (char*)malloc(113*200*3*sizeof(char));
    for (int i=0; i<113*200*3; i++) {
        int res = (int)(*(predOutput2+i) + 128);
        if (res > 255)
            res = 255;
        if (res < 0)
            res = 0;
        *(p2+i) = res;
    }
    
    cv::Mat mat0(452, 800, CV_8UC3, (void*)p0);
    cv::imwrite("../output/0.jpg", mat0);

    cv::Mat mat1(226, 400, CV_8UC3, (void*)p1);
    cv::imwrite("../output/1.jpg", mat1);

    cv::Mat mat2(113, 200, CV_8UC3, (void*)p2);
    cv::imwrite("../output/2.jpg", mat2);
}

int main() {
    m_task = std::move(std::unique_ptr<snpetask::SNPETask>(new snpetask::SNPETask()));

    std::string model_path = std::string("../model/modified_deepdeblur_GOPRO-L1_snpe-2.13_quantize_cached_v68.dlc");
    runtime_t runtime = runtime::DSP;
    inputLayers = {"0", "1", "2"};
    outputLayers = {"Conv_245", "Conv_164", "Conv_83"};
    outputTensors = {"492", "411", "330"};

    m_task->setOutputLayers(outputLayers);
    if (!m_task->init(model_path, runtime)) {
        printf("Can't init snpetask instance.\n");
        return false;
    }

    std::string img_path("../input/Istanbul_blur1.png");
    cv::Mat img = cv::imread(img_path);
    PreProcess(img);
    auto start = std::chrono::high_resolution_clock::now();
    m_task->execute();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Exec " << duration.count()/1000.0 << "ms" << std::endl;
    PostProcess();

    return 0;
}
