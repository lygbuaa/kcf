#ifndef APCE_CRITERION_H
#define APCE_CRITERION_H

#include <opencv2/opencv.hpp>
#include <deque>

class ApceCriterion
{
public:
    ApceCriterion(int slide_win_len = 30, double apce_thresh_coef = 0.3f);
    ~ApceCriterion();

    void update(const cv::Mat& scores_map);
    void reset();
    bool judge() const;
    double getApce() const;

private:
    const int SLIDE_WIN_LEN_;
    const double APCE_THRESH_COEF_;
    std::deque<double> apceQueue_;
    double apce_;
    double avgApce_;
};


#endif //APCE_CRITERION_H
