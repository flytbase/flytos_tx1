// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file 
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and


//ROS related includes
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <sstream>
using namespace caffe;  // NOLINT(build/namespaces)

std::string class_labels[] = {"__background__","Aeroplane","Bicycle","Bird","Boat","Bottle", "Bus", "Car", "Cat", "Chair","Cow", "Diningtable", "Dog", "Horse","Motorbike", "Person", "Foliage","Sheep", "Sofa", "Train", "Tvmonitor"};
cv_bridge::CvImagePtr new_img_ptr;
std_msgs::Header new_img_header;
std_msgs::Header current_img_header;
cv::Scalar white(255,255,255);

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{ 
  try
  {
    new_img_header = msg->header;
    new_img_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

class Detector 

{
public:
  Detector(const string& model_file,
   const string& weights_file,
   const string& mean_file,
   const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
    std::vector<cv::Mat>* input_channels);

private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
 const string& weights_file,
 const string& mean_file,
 const string& mean_value) 
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
  << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) 
{
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
   input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) 
  {
    if (result[0] == -1) 
    {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) 
{
  cv::Scalar channel_mean;
  if (!mean_file.empty()) 
  {
    CHECK(mean_value.empty()) <<
    "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) 
    {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) 
  {
    CHECK(mean_file.empty()) <<
    "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) 
    {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
    "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) 
    {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
        cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
 void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
 {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) 
  {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
  std::vector<cv::Mat>* input_channels) 
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
  //cv::imshow("resized",sample_resized);
  //cv::waitKey(1);
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
   cv::split(sample_normalized, *input_channels);

   CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
    == net_->input_blobs()[0]->cpu_data())
   << "Input channels are not wrapping the input layer of the network.";
 }

 DEFINE_string(mean_file, "",
  "The mean file used to subtract from the input image.");
 DEFINE_string(mean_value, "104,117,123",
  "If specified, can be one value or can be same as image channels"
  " - would subtract from the corresponding channel). Separated by ','."
  "Either mean_file or mean_value should be provided, not both.");
 DEFINE_string(file_type, "image",
  "The file type in the list_file. Currently support image and video.");
 DEFINE_string(out_file, "",
  "If provided, store the detection results in the out_file.");
 DEFINE_double(confidence_threshold, 0.01,
  "Only store detections with score higher than the threshold.");

////////////////////////////////////////////////////
////////////////////main loop///////////////////////
////////////////////////////////////////////////////

 int main(int argc, char** argv) 
 {

  ros::init(argc, argv, "ssd_detect");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe("/flytos/flytcam/image_raw", 1, imageCallback);
  image_transport::Publisher image_pub = it.advertise("/flytos/flytcam/detected_objects", 1);
  ros::AsyncSpinner spinner(2); // Use 4 threads
  spinner.start();


  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
  #ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
  #endif
  gflags::SetUsageMessage("Do detection using SSD mode.\n"
    "Usage:\n"
    "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 3) 
  {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  while(!new_img_ptr && ros::ok())
 {
  ROS_INFO("Waiting for data on /flytos/usb_cam/image_raw topic");
  sleep(1);
 }

  while (ros::ok()) 
  {
      ssd_caffe::Detections detections_msg;
      //current_img_header = new_img_header;
      detections_msg.header = new_img_header;
      cv::Mat img_uncropped = new_img_ptr->image;
	    cv::Mat img = img_uncropped(cv::Rect((int(img_uncropped.cols - img_uncropped.rows)/2),0,img_uncropped.rows,img_uncropped.rows));
      std::vector<vector<float> > detections = detector.Detect(img);

      for (int i = 0; i < detections.size(); ++i) 
      {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);   
        const float score = d[2];
        if (score >= confidence_threshold) 
        {
        	//print the bbox and label text on the img
          std::ostringstream ostr_c; //output string stream
          ostr_c.precision(2);
          ostr_c << std::fixed;
          ostr_c << (score * 100);
          std::string confidence_text = ostr_c.str() + "%";
          
          cv::Mat roi = img(cv::Rect(static_cast<int>(d[3] * img.cols), static_cast<int>(d[4] * img.rows), static_cast<int>((d[5]-d[3]) * img.cols) ,static_cast<int>((d[6]-d[4]) * img.rows) ));
          cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(255, 0, 0)); 
          
          cv::addWeighted(color, 0.5, roi, 0.5 , 0.0, roi);
        	cv::putText(img,class_labels[int(d[1])],cv::Point(static_cast<int>(d[3] * img.cols),static_cast<int>(d[4] * img.rows) +25), cv::FONT_HERSHEY_TRIPLEX,0.8,white,1,8);
          cv::putText(img,confidence_text,cv::Point(static_cast<int>(d[3] * img.cols),static_cast<int>(d[4] * img.rows) +50), cv::FONT_HERSHEY_TRIPLEX,0.8,white,1,8);
        }
      }
      sensor_msgs::ImagePtr pub_msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8", img).toImageMsg();
      image_pub.publish(pub_msg);
  }
  ros::waitForShutdown();
  return 0;
}

