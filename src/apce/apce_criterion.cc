#include "apce_criterion.h"

ApceCriterion::ApceCriterion(int SLIDE_WIN_LEN_, double apce_thresh_coef)
: SLIDE_WIN_LEN_(SLIDE_WIN_LEN_), APCE_THRESH_COEF_(apce_thresh_coef) 
{
    reset();
}

ApceCriterion::~ApceCriterion() {}

void ApceCriterion::reset()
{
    apceQueue_.clear();
    apce_ = 0.0f;
    avgApce_ = 0.0f;
}

void ApceCriterion::update(const cv::Mat& scores_map)
{
    double min_value, max_value;
    cv::minMaxLoc(scores_map, &min_value, &max_value);

    cv::Mat scores_offset = scores_map - min_value;
    cv::Mat scores_offset_sqr = scores_offset.mul(scores_offset);
    cv::Scalar sum_sqr = cv::sum(scores_offset_sqr);
    double mean_amp = (double)sum_sqr[0] / (double) scores_offset.total();
    apce_ = (double)(pow(max_value - min_value, 2) / (mean_amp + 1e-7));

    if(judge())
    {
        size_t count = apceQueue_.size();
        if(count == 0){
            avgApce_ = apce_;
        }else if(count < SLIDE_WIN_LEN_){
            avgApce_ = (avgApce_ * count + apce_) / (count + 1);
        }
        else{
            double front_apce = apceQueue_.front();
            avgApce_ = (avgApce_ * SLIDE_WIN_LEN_ - front_apce + apce_) / SLIDE_WIN_LEN_;
            apceQueue_.pop_front();
        }
        apceQueue_.push_back(apce_);
    }
}

bool ApceCriterion::judge() const
{
    const double apce_thresh = APCE_THRESH_COEF_ * avgApce_;
    return (apce_ >= apce_thresh);
}

double ApceCriterion::getApce() const
{
    return apce_;
}

