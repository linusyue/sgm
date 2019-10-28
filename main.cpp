#include <iostream>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "semi_global_matching.h"
#include "evaluate_disp.h"

#include <dirent.h>

using namespace cv;
using namespace std;

void usage(char* argv[])
{
    //Folder struct
    //-directory/
    //     --image2/
    //     --image3/
    std::cout << "usage: " << argv[0] << "sgm [directory]" << std::endl;

    cout<<"Foler structure"<<endl;
    cout<<"-[directory]/"<<endl;
    cout<<"      --image2/"<<endl;
    cout<<"      --image3/"<<endl;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        usage(argv);
        return -1;
    }

    string directory = argv[1];

    DIR *dp, *dp1;
    struct dirent *ep;

    string image2_dir;
    image2_dir = directory + "/" + "image_2";
    dp = opendir(image2_dir.c_str());

    string image3_dir;
    image3_dir = directory + "/" + "image_3";
    dp1 = opendir(image3_dir.c_str());

    if (dp == NULL || dp1 ==  NULL) {
        std::cerr << "Invalid folder structure under: " << directory << std::endl;
        usage(argv);
        exit(EXIT_FAILURE);
    }

    string I0_path;
    string I1_path;

    SemiGlobalMatching::Parameters param;
    SemiGlobalMatching sgm(param);
    cv::Mat D0, D1, draw;
   
    system("mkdir -p results/data/disp_0/");

    while ((ep = readdir(dp)) != NULL) 
    {
        // Skip directories
        if (!strcmp (ep->d_name, "."))
            continue;
        if (!strcmp (ep->d_name, ".."))
            continue;

        string postfix = "_10.png";
        string::size_type idx;

        string image_name = ep->d_name;
        
        //Only _10 has groundtruth
        idx = image_name.find(postfix);
        
        if(idx == string::npos )
            continue;  

        I0_path = directory + "/" + "image_2" + "/" + image_name;
        I1_path = directory + "/" + "image_3" + "/" + image_name;

        cout<<"I0: "<<I0_path<<endl;
        cout<<"I1: "<<I1_path<<endl;

        Mat I0 = imread(I0_path);
        Mat I1 = imread(I1_path);

        if (I0.empty() || I1.empty())
        {
            std::cerr << "failed to read any image." << std::endl;
            break;
        }

        CV_Assert(I0.size() == I1.size() && I0.type() == I1.type());

        //convert to gray
        Mat I0_Gray, I1_Gray;
        cvtColor(I0, I0_Gray, cv::COLOR_BGR2GRAY);
        cvtColor(I1, I1_Gray, cv::COLOR_BGR2GRAY);

        imshow("I0", I0);
        imshow("I1", I1);

        const auto t1 = std::chrono::system_clock::now();

        sgm.compute(I0_Gray, I1_Gray, D0, D1);

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;

        D0.convertTo(draw, CV_8U, 255. / (SemiGlobalMatching::DISP_SCALE * param.numDisparities));
        cv::applyColorMap(draw, draw, cv::COLORMAP_JET);
        draw.setTo(0, D0 == SemiGlobalMatching::DISP_INV);

        cv::imshow("disparity", draw);

        Mat D0_16u(D0.size(), CV_16U);

        //convert to kitti format
        for(int i=0; i<D0.rows; i++)
        {
            for(int j=0; j<D0.cols; j++)
            {
                int d = D0.at<int16_t>(i, j);

                if (d<0) d=0;

                D0_16u.at<uint16_t>(i, j) = d*256/(float)SemiGlobalMatching::DISP_SCALE;
            }
        }

        //Saving result
        imwrite(("results/data/disp_0/" + image_name).c_str(), D0_16u);

        eval(directory, image_name);

        waitKey();
    }

    return 0; 
}
