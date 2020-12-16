#include <ros/ros.h>
#include "dataset_filter/DataSetFilter.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dataset_filter_col");
  ros::NodeHandle nodeHandle("~");

  dataset_filter::DataSetFilter DataSetFilter(nodeHandle);

  ros::spin();
  return 0;
}
