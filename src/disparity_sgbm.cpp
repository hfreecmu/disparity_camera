
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <image_transport/image_transport.h>


//TODO reconcile intrinsic functions in this and pouintcloud_node.cpp into common file
struct Intrinsics
{
    float baseline;
    float f_norm;
    float cx;
    float cy;
};

std::shared_ptr<Intrinsics> getCameraInfo(const sensor_msgs::CameraInfoConstPtr &rightCameraInfo)
{
    //TODO is this right combining left and right?
    //is left f_norm and right f_norm the same?
    //which should be used when calculating baseline?

    float baseline = rightCameraInfo->P[3] / rightCameraInfo->P[0];
    float f_norm = rightCameraInfo->P[0];
    float cx = rightCameraInfo->P[2];
    float cy = rightCameraInfo->P[6];

    std::shared_ptr<Intrinsics> intrinsics = std::make_shared<Intrinsics>();
    intrinsics->baseline = baseline;
    intrinsics->f_norm = f_norm;
    intrinsics->cx = cx;
    intrinsics->cy = cy;

    return intrinsics;
}

void inpaintImage(const cv::Mat& disparityImage,
                  cv::Mat& inpaintedDisparityImage,
                  int minDisparity)
{
    cv::Mat mask = cv::Mat::zeros(disparityImage.size(), CV_8UC1);
    mask.setTo(1, disparityImage == (minDisparity - 1));
    
    //TODO this is not fully finished so adding ros warn.
    //Also requires cv2 3.4.2 which you can install following
    //https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
    //but I ran into lib conflicts for wls filter and so got rid of it
    ROS_WARN_STREAM("Inpainting should not consider parts outside maxDisparity.");
    cv::inpaint(disparityImage, mask, inpaintedDisparityImage, 1.0, 0);
}

class DisparityAction 
{
public:
DisparityAction(ros::NodeHandle &nh_, std::string disparityTopic, int disparityQueueSize,
                int minDisparity, int numDisparities,
                int blockSize,
                int P1, int P2, 
                int disp12MaxDiff,
                int preFilterCap,
                int uniquenessRatio,
                int speckleWindowSize,
                int speckleRange,
                int mode,
                bool inpaint,
                bool wlsFilter,
                double wlsLambda,
                double wlsSigma,
                bool fbs

):cv_ptr(new cv_bridge::CvImage),
sgbm(cv::StereoSGBM::create(minDisparity, numDisparities, blockSize)),
inpaint(inpaint),
wlsFilter(wlsFilter),
fbs(fbs),
minDisparity(minDisparity)
{
    disparityPub = nh_.advertise<sensor_msgs::Image>(disparityTopic,  disparityQueueSize);

    sgbm->setMinDisparity(minDisparity);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setBlockSize(blockSize);
    sgbm->setDisp12MaxDiff(disp12MaxDiff);
    sgbm->setPreFilterCap(preFilterCap);
    sgbm->setUniquenessRatio(uniquenessRatio);
    sgbm->setSpeckleWindowSize(speckleWindowSize);
    sgbm->setSpeckleRange(speckleRange);

    if (mode == 0)
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    else if (mode == 1)
        sgbm->setMode(cv::StereoSGBM::MODE_HH);
    else if (mode == 2)
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    else if (mode == 3)
        sgbm->setMode(cv::StereoSGBM::MODE_HH4);
    else
    {
        ROS_WARN("Illegal mode, using sgbm default");
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    }

    if (wlsFilter)
    {
        disparityWLSFilter = cv::ximgproc::createDisparityWLSFilter(sgbm);
        rightSgbm = cv::ximgproc::createRightMatcher(sgbm);

        disparityWLSFilter->setLambda(wlsLambda);
        disparityWLSFilter->setSigmaColor(wlsSigma);
    }
}

void sgbm_compute(const sensor_msgs::ImageConstPtr &leftImage, 
                  const sensor_msgs::CameraInfoConstPtr &leftCameraInfo,
                  const sensor_msgs::ImageConstPtr &rightImage, 
                  const sensor_msgs::CameraInfoConstPtr &rightCameraInfo
){
    //Get camera info
    if (cameraInfo == nullptr)
        cameraInfo = getCameraInfo(rightCameraInfo);

    //bgr or rgb?
    cv::Mat left = cv_bridge::toCvCopy(*leftImage, "bgr8")->image;
    cv::Mat right = cv_bridge::toCvCopy(*rightImage, "bgr8")->image;

    cv::Mat grayLeft, grayRight;
    cvtColor(left, grayLeft, cv::COLOR_BGR2GRAY);
    cvtColor(right, grayRight, cv::COLOR_BGR2GRAY);
 
    cv::Mat left_for_matcher = grayLeft.clone();
    cv::Mat right_for_matcher = grayRight.clone();

    //sgbm
    cv::Mat sgbm_disp;
    cv::Mat disp8(left.rows, left.cols, CV_32F);

    sgbm->compute(left_for_matcher, right_for_matcher, sgbm_disp);

    cv::Mat right_sgbm_disp;
    cv::Mat filtered_disp;
    cv::Mat fbs_disp;
    if (!wlsFilter)
    {
        if (!fbs)
            sgbm_disp.convertTo(disp8, CV_32F, 1.0/16.0);
        else
        {
            cv::Mat bfDisp8;
            sgbm_disp.convertTo(bfDisp8, CV_32F, 1.0/16.0);
            cv::bilateralFilter(bfDisp8, disp8, 5, 10, 10);
        }
    }
    else
    {
        rightSgbm->compute(right_for_matcher, left_for_matcher, right_sgbm_disp);
        disparityWLSFilter->filter(sgbm_disp, left_for_matcher, filtered_disp, right_sgbm_disp);
        if (fbs)
        {
            cv::Mat conf_map = disparityWLSFilter->getConfidenceMap();
            cv::ximgproc::fastBilateralSolverFilter(grayLeft, filtered_disp, conf_map/255.0f, fbs_disp);
            fbs_disp.convertTo(disp8, CV_32F, 1.0/16.0);
        }
        else
            filtered_disp.convertTo(disp8, CV_32F, 1.0/16.0);
    }

    if (inpaint)
    {
        cv::Mat inpaintedDisp8 = cv::Mat::zeros(disp8.size(), CV_32F);
        inpaintImage(disp8, inpaintedDisp8, minDisparity);
        disp8 = inpaintedDisp8;
    }

    //convert to ros message
    cv_ptr->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_ptr->header.stamp = leftImage->header.stamp;
    cv_ptr->header.frame_id = leftImage->header.frame_id;
    cv_ptr->image = disp8;

    sensor_msgs::Image imageToPublish;
    cv_ptr->toImageMsg(imageToPublish);

    disparityPub.publish(imageToPublish);

    ROS_INFO("Stereo Image Succesfully Processed ...............................");
}

~DisparityAction(){}

protected:
bool inpaint;
bool wlsFilter;
bool fbs;
int minDisparity;
std::shared_ptr<Intrinsics> cameraInfo;
cv::Ptr<cv::StereoSGBM> sgbm;
cv_bridge::CvImagePtr cv_ptr;
cv::Ptr<cv::StereoMatcher> rightSgbm;
cv::Ptr<cv::ximgproc::DisparityWLSFilter> disparityWLSFilter;
ros::Publisher disparityPub;
};

int main(int argc, char** argv){

    //init ros node
    ros::init(argc, argv, "disparity_sgbm");

    //create async spinner
    ros::AsyncSpinner spinner(6);
    spinner.start();

    //create node handles
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    //ros params
    int minDisparity;
    int numDisparities;
    int blockSize;
    int P1;
    int P2;
    int disp12MaxDiff;
    int preFilterCap;
    int uniquenessRatio;
    int speckleWindowSize;
    int speckleRange;
    int mode;
    bool inpaint;
    bool wlsFilter;
    double wlsLambda;
    double wlsSigma;
    bool fbs;

    bool success = true;
    success &= nhp.getParam("min_disparity", minDisparity);
    success &= nhp.getParam("num_disparities", numDisparities);
    success &= nhp.getParam("block_size", blockSize);
    success &= nhp.getParam("P1", P1);
    success &= nhp.getParam("P2", P2);
    success &= nhp.getParam("disp12_max_diff", disp12MaxDiff);
    success &= nhp.getParam("pre_filter_cap", preFilterCap);
    success &= nhp.getParam("uniqueness_ratio", uniquenessRatio);
    success &= nhp.getParam("speckle_window_size", speckleWindowSize);
    success &= nhp.getParam("speckle_range", speckleRange);
    success &= nhp.getParam("sgbm_mode", mode);
    success &= nhp.getParam("inpaint", inpaint);
    success &= nhp.getParam("wls_filter", wlsFilter);
    success &= nhp.getParam("wls_lambda", wlsLambda);
    success &= nhp.getParam("wls_sigma", wlsSigma);
    success &= nhp.getParam("fbs", fbs);

    if (!success)
    {
        ROS_WARN("Failed to read parameters");
        return 0;
    }

    //subscriptions
    std::string leftImageTopic = "/theia/left/image_rect_color";
    std::string leftInfoTopic = "/theia/left/camera_info";
    std::string rightImageTopic = "/theia/right/image_rect_color";
    std::string rightInfoTopic = "/theia/right/camera_info";

    message_filters::Subscriber<sensor_msgs::Image> leftImageSub(nh, leftImageTopic, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> leftInfoSub(nh, leftInfoTopic, 1);
    message_filters::Subscriber<sensor_msgs::Image> rightImageSub(nh, rightImageTopic, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> rightInfoSub(nh, rightInfoTopic, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;

    MySyncPolicy mySyncPolicy(9);
    mySyncPolicy.setAgePenalty(1.0);
    mySyncPolicy.setInterMessageLowerBound(ros::Duration(0.2));
    mySyncPolicy.setMaxIntervalDuration(ros::Duration(0.1));

    const MySyncPolicy myConstSyncPolicy = mySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(myConstSyncPolicy, leftImageSub, leftInfoSub, rightImageSub, rightInfoSub);


    DisparityAction disparityNode(nh, "/disparity_image", 1,
                                  minDisparity, numDisparities,
                                  blockSize,
                                  P1, P2,
                                  disp12MaxDiff,
                                  preFilterCap,
                                  uniquenessRatio,
                                  speckleWindowSize,
                                  speckleRange,
                                  mode,
                                  inpaint,
                                  wlsFilter,
                                  wlsLambda,
                                  wlsSigma,
                                  fbs
    );

    sync.registerCallback(boost::bind(&DisparityAction::sgbm_compute, &disparityNode, _1, _2, _3, _4));

    ROS_INFO("SGBM Disparity Node Started");

    ros::waitForShutdown();

    return 0;
}