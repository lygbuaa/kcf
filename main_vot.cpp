#include <stdlib.h>

#include "kcf.h"
#include "vot.hpp"

int main()
{
    static const std::string DATA_PATH = "/home/hugoliu/hugo/github/kcf/data/";
    static const std::string REGION_FILE_PATH = DATA_PATH + "region.txt";
    static const std::string IMAGES_FILE_PATH = DATA_PATH + "images.txt";

    //load region, images and prepare for output
    VOT vot_io(REGION_FILE_PATH, IMAGES_FILE_PATH, "output.txt");

    KCF_Tracker tracker;
    cv::Mat image;

    //img = firts frame, initPos = initial position in the first frame
    cv::Rect init_rect = vot_io.getInitRectangle();
    vot_io.outputBoundingBox(init_rect);
    vot_io.getNextImage(image);

    tracker.init(image, init_rect);

    BBox_c bb;
    double avg_time = 0.;
    int frames = 0;
    cv::namedWindow("kcf", cv::WINDOW_NORMAL);

    while (vot_io.getNextImage(image) == 1){
        double time_profile_counter = cv::getCPUTickCount();
        tracker.track(image);
        time_profile_counter = cv::getCPUTickCount() - time_profile_counter;
        //std::cout << "  -> speed : " <<  time_profile_counter/((double)cvGetTickFrequency()*1000) << "ms. per frame" << std::endl;
        avg_time += time_profile_counter/((double)cv::getTickFrequency()/1000.0f);
        frames++;

        bb = tracker.getBBox();
        vot_io.outputBoundingBox(cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h));
        double score = tracker.getMaxResponse();
        char buf[64] = {0};
        sprintf(buf, "score: %.2f", score);

        // std::cout << "frame: " << frames << ", bbox: " << bb.cx << ", " << bb.cy << ", " << bb.w << ", " << bb.h << std::endl;

        cv::rectangle(image, cv::Rect(bb.cx - bb.w/2., bb.cy - bb.h/2., bb.w, bb.h), CV_RGB(0,255,0), 2);
        cv::putText(image, std::string(buf), cv::Point(bb.cx-bb.w/2, bb.cy-bb.h/2-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 64, 0), 2);
        cv::imshow("kcf", image);
        cv::waitKey(30);

//        std::stringstream s;
//        std::string ss;
//        int countTmp = frames;
//        s << "imgs" << "/img" << (countTmp/10000);
//        countTmp = countTmp%10000;
//        s << (countTmp/1000);
//        countTmp = countTmp%1000;
//        s << (countTmp/100);
//        countTmp = countTmp%100;
//        s << (countTmp/10);
//        countTmp = countTmp%10;
//        s << (countTmp);
//        s << ".jpg";
//        s >> ss;
//        //set image output parameters
//        std::vector<int> compression_params;
//        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
//        compression_params.push_back(90);
//        cv::imwrite(ss.c_str(), image, compression_params);
    }

    std::cout << "Average processing speed " << avg_time/frames <<  "ms. (" << 1./(avg_time/frames)*1000 << " fps)" << std::endl;

    return EXIT_SUCCESS;
}