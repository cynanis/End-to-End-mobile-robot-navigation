#include "dataset_filter/DataSetFilter.hpp"
#include <cmath>
#include<std_msgs/Header.h>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>


namespace dataset_filter
{

  DataSetFilter::DataSetFilter(ros::NodeHandle &nodeHandle) : nodeHandle_(nodeHandle)
  {
    //initialize the publisher
    sync_scan_pub_ = nodeHandle_.advertise<sensor_msgs::LaserScan>("/laser_scan_sync", 100);
    sync_cmd_vel_pub_ = nodeHandle_.advertise<geometry_msgs::TwistStamped>("/cmd_vel_synced", 100);
    sync_goals_up = nodeHandle_.advertise<geometry_msgs::PoseStamped>("/goals_up_synced", 100);


    //sync_pcl          = nodeHandle_.advertise<sensor_msgs::PointCloud>("/sync_pcl",10);
    //sync_image          = nodeHandle_.advertise<sensor_msgs::Image>("/sync_image",10);

    //initialize the tf filter
    laser_sub_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nodeHandle_, "/scan", 100);
    cmd_vel_sub = new message_filters::Subscriber<geometry_msgs::TwistStamped>(nodeHandle_, "/cmd_vel_stamped", 30);
    goal_sub = new message_filters::Subscriber<geometry_msgs::PoseStamped>(nodeHandle_, "/goals_sub", 100);
    //initialize the tf filter
    laser_tf_filter_ = new tf::MessageFilter<sensor_msgs::LaserScan>(*laser_sub_, listener_laser, "base_link", 100);
    //cmd_vel_tf_filter_ = new tf::MessageFilter<geometry_msgs::TwistStamped>(cmd_vel_sub,listener_, "base_link", 5);
    goal_tf_filter = new tf::MessageFilter<geometry_msgs::PoseStamped>(*goal_sub, listener_goals, "base_link", 100);


    sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(1000), *laser_tf_filter_, *cmd_vel_sub, *goal_tf_filter);
    sync->registerCallback(boost::bind(&DataSetFilter::syncedCallback, this, _1, _2, _3));
  }

  void DataSetFilter::syncedCallback(const sensor_msgs::LaserScan::ConstPtr &scan_in,
                                     const geometry_msgs::TwistStamped::ConstPtr &cmd_vel_in,
                                     const geometry_msgs::PoseStamped::ConstPtr &goals_in)
  {

    
    geometry_msgs::PoseStamped goals_out;

    //sensor_msgs::LaserScan scan_out;
    //scan_out = *scan_in;
    //printf("before bug............................\n");
    //scan_out.ranges[1] = 1.25;
    //printf("after bug--------------------------\n");

    try
    {

      listener_goals.transformPose(target_frame, *goals_in, goals_out);
  

      //transformLaser(target_frame, *scan_in, scan_out);
    }
    catch (tf::TransformException &ex)
    {

      printf("Failure %s\n", ex.what()); //Print exception which was caught
    }

    sync_scan_pub_.publish(scan_in);
    sync_cmd_vel_pub_.publish(cmd_vel_in);
    sync_goals_up.publish(goals_out);

  }

  void DataSetFilter::transformLaser(std::string target_frame, sensor_msgs::LaserScan scan_in,sensor_msgs::LaserScan &scan_out)
  {
    int range_size = (int)((scan_in.angle_max - scan_in.angle_min) / scan_in.angle_increment);
    float X_, Y_,theta,ranges[range_size];
    geometry_msgs::PointStamped point_in,point_out;
    point_in.header = scan_in.header;
    scan_out.ranges[3]=2.35;
    for (int i = 1; i <= range_size; i++)
    {

      if (scan_in.ranges[i]<scan_in.range_max and scan_in.ranges[i]> scan_in.range_min)
      {
        //get xy point coordinates w.r.t the robot pose
        getXY(scan_in, X_, Y_, i);
        point_in.point.x = X_;
        point_in.point.y = Y_;
        point_in.point.z = 0.0;
        listener_laser.transformPoint(target_frame,point_in,point_out);
        ROS_INFO_STREAM( "transformed point (x,y,z) : " <<"("<< point_out.point.x <<","<<point_out.point.y<<","<<point_out.point.z<<")");
        getThetaR(point_out,theta,scan_out.ranges[i]);
        ROS_INFO_STREAM( "range : " <<ranges[i]<<" theta: "<<theta);
        //scan_out.ranges[i] = ranges[i];
        ROS_INFO_STREAM( "assigned range : "<< scan_out.ranges[i]);
        
      }
    
    }
    
    scan_out.header = point_out.header;
    scan_out.angle_min = scan_in.angle_min;
    scan_out.angle_max = scan_in.angle_max;
    scan_out.angle_increment = scan_in.angle_increment;
    scan_out.range_max = scan_in.range_max;
    scan_out.range_min = scan_in.range_min;
    scan_out.scan_time = scan_in.scan_time;
    
  }
  void DataSetFilter::getXY(sensor_msgs::LaserScan scan_in, float &X, float &Y, int i)
  {
    float angle = (scan_in.angle_min + i * scan_in.angle_increment);
    if (angle > M_PI)
    {
      angle = (2 * M_PI) - angle;
    }
    else if (angle < -M_PI)
    {
      angle = (2 * M_PI) + angle;
    }
    X = scan_in.ranges[i] * cos(angle);
    Y = scan_in.ranges[i] * sin(angle);
  }
  void DataSetFilter::getThetaR(geometry_msgs::PointStamped point,float& theta,float& range){
      range = (float)sqrt(std::pow(point.point.x,2)+std::pow(point.point.y,2));
      theta = (float)atan(point.point.y/range);


  }
  DataSetFilter::~DataSetFilter()
  {
  }

} // namespace dataset_filter
