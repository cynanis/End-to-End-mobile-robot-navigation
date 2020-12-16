#include <ros/ros.h>

#include "tf/transform_listener.h"
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

//#include "laser_geometry/laser_geometry.h"

#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
//#include <sensor_msgs/PointCloud.h>


namespace dataset_filter {

    class DataSetFilter
    {

        public:
        
        DataSetFilter(ros::NodeHandle &nodeHandle);




        virtual ~DataSetFilter();

        private:
        void syncedCallback(const sensor_msgs::LaserScan::ConstPtr& scan_in, 
                            const geometry_msgs::TwistStamped::ConstPtr& cmd_vel_in,
                            const geometry_msgs::PoseStamped::ConstPtr& goals_in);


        // void getXY(const sensor_msgs::LaserScan::ConstPtr & scan_in, float &X, float &Y, int i);
        


        void transformLaser(std::string target_frame,sensor_msgs::LaserScan scan_in,sensor_msgs::LaserScan &scan_out);
        void getXY(sensor_msgs::LaserScan scan_in, float &X, float &Y, int i);
        void getThetaR(geometry_msgs::PointStamped point,float& theta,float& range);




        ros::NodeHandle &nodeHandle_;
        // tf listeners
        tf::TransformListener listener_laser,listener_goals,listener_pose;
        std::string target_frame= "base_link";


        
        //message filters
        message_filters::Subscriber<sensor_msgs::LaserScan>* laser_sub_;
        message_filters::Subscriber<geometry_msgs::TwistStamped>* cmd_vel_sub;
        message_filters::Subscriber<geometry_msgs::PoseStamped>* goal_sub;

        

        //tf filters
        tf::MessageFilter<sensor_msgs::LaserScan>* laser_tf_filter_;
        //tf::MessageFilter<geometry_msgs::TwistStamped>* cmd_vel_tf_filter_;
        tf::MessageFilter<geometry_msgs::PoseStamped>* goal_tf_filter;

        

        //Synchronizer
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::LaserScan,geometry_msgs::TwistStamped,geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy>* sync;

        //synchronized topic publisher 
        ros::Publisher sync_scan_pub_;
        ros::Publisher sync_cmd_vel_pub_;
        ros::Publisher sync_goals_up;
  

        
        
    };
}
