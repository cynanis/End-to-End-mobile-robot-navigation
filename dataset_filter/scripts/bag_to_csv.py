#!/usr/bin/env python

from __future__ import print_function
import rospy
import rosbag
import cv2
import os
import sys
import time
import csv
import numpy as np
import pandas as pd
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


class Bag_TO_CSV:

    def __init__(self, bag_file_dir=None, root_save=None):
        self.bag_file_dir = bag_file_dir
        self.root_save = root_save
        self.scan_topic = "/laser_scan_sync"
        self.cmd_vel_topic = "/cmd_vel_synced"
        self.goals_topic = "/goals_up_synced"

        self.image_name = []
        self.cmd_vel_data = []
        self.goal_data = []
        self.scan_data = []
        self.save_step = 0
        self.index = 1
        self.save_size = 400
        self.bagToCSV()

    def bagToCSV(self):
        # get list of only bag files in current dir.
        listOfBagFiles = [f for f in os.listdir(
            self.bag_file_dir) if f[-4:] == ".bag"]

        numberOfFiles = str(len(listOfBagFiles))
        print("found ", numberOfFiles, "of bag files")
        for f in listOfBagFiles:
            print(f)
            print(
                "\n press ctrl+c in the next  second to cancel \n")
            time.sleep(1)

        # loop through the bagfiles
        bagNumber = 0
        with open(self.root_save+"/CSV/image_name.csv", 'w') as image_name_file,  open(self.root_save+"/CSV/ranges.csv", 'w') as ranges_file, open(self.root_save+"/CSV/command_velocities.csv", 'w') as cmd_vel_file, open(self.root_save+"/CSV/goals.csv", 'w') as goals_file:
            Image_name_Writer = csv.writer(image_name_file, delimiter=',')
            range_Writer = csv.writer(ranges_file, delimiter=',')
            cmd_vel_Writer = csv.writer(cmd_vel_file, delimiter=',')
            goals_Writer = csv.writer(goals_file, delimiter=',')

            # setting up the header of the  data
            Image_name_Writer.writerow(['imageName'])
            cmd_vel_Writer.writerow(['linearX',  'angluarV'])
            goals_Writer.writerow(['goalX', 'goalY', 'goalHeading'])

            range_headers = ["range"+str(j)
                             for j in np.arange(start=1, stop=361)]

            range_Writer.writerow(range_headers)
            scandx, cmddx, goaldx, posedx = 0, 0, 0, 0

            for bagFile in listOfBagFiles:
                bagNumber += 1
                print("reading file " + str(bagNumber) +
                      " of  " + str(numberOfFiles) + ": " + bagFile)
                # access bag
                bag = rosbag.Bag(self.bag_file_dir+"/"+bagFile)
                bagContents = bag.read_messages()

                print("extracting the bag number", bagNumber, "information")

                for topic, msg, t in bagContents:
                    if topic == self.scan_topic:
                        scandx += 1
                        # tscan = True
                        image_name = "image"+str(self.index)+".jpg"
                        # Image_name_Writer.writerow([image_name])

                        scan_data = np.array(list(msg.ranges))
                        super_threshold_indices = np.isinf(scan_data)
                        scan_data[super_threshold_indices] = -1.0
                        scan_data = scan_data.tolist()
                        range_Writer.writerow(scan_data)

                    elif topic == self.cmd_vel_topic:
                        cmddx += 1
                        cmd_angluar = msg.twist.angular.z
                        cmd_x = msg.twist.linear.x
                        cmd_y = msg.twist.linear.y
                        #self.cmd_vel_data = [cmd_x, cmd_y, cmd_angluar]

                        cmd_vel_Writer.writerow([cmd_x, cmd_angluar])
                        #tcmd = True
                    elif topic == self.goals_topic:
                        goaldx += 1
                        quaternion = (
                            msg.pose.orientation.x,
                            msg.pose.orientation.y,
                            msg.pose.orientation.z,
                            msg.pose.orientation.w)
                        euler = euler_from_quaternion(quaternion)
                        goal_heading = euler[2]
                        goal_x = msg.pose.position.x
                        goal_y = msg.pose.position.y
                        goals_Writer.writerow([goal_x, goal_y, goal_heading])
                bag.close()

# convert 2d point clouds to images if needed
    def rangesToImage(self, msg, image_name, pose_msg):
        num_scans = len(msg.ranges)

        img_dim = num_scans*2
        # initialize white image
        image = np.full((img_dim, img_dim), 255, dtype="uint8")

        # heading and position of the robot in the image
        pose_x, pose_y, pose_theta = pose_msg.pose.position.x, pose_msg.pose.position.y, euler_from_quaternion((
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w))[2]
        pose_x, pose_y = int((img_dim/2.0)+pose_x), int((img_dim/2.0)-pose_y)
        # felling the image with robot pose
        image[pose_y-2:pose_y+2, pose_x-2:pose_x+2] = 0
        if pose_theta <= (np.pi/2) and pose_theta >= (-np.pi/2):
            npose_x = pose_x+3
            npose_y = int(pose_theta * 3 + pose_y+2)
            image[npose_y:npose_y, npose_x:npose_x] = 0
        else:
            npose_x = pose_x-3
            npose_y = int(pose_theta * (-3) + pose_y+2)
            image[npose_y:npose_y, npose_x:npose_x] = 0
        # Fill the image with laser scan
        for i, range in enumerate(msg.ranges, 1):
            if range <= msg.range_max and range >= msg.range_min:
                # convert polar to cartisian cordinates
                x, y = self.getXY(range, msg.angle_min, msg.angle_increment, i)

                # scal the xy coordinates to the image size
                # print(num_scans/msg.range_max)
                x, y = x*num_scans/msg.range_max, y * num_scans/msg.range_max
                # shift the origin (laser origin) to the top left corner of the image
                x, y = x+(img_dim/2.0), (img_dim/2.0)-y
                x, y = int(x), int(y)
                if y > img_dim:
                    print("problem is here ************", y, ">", img_dim)
                # //print("x,y", x, y)
                # print("[X,Y]=[{},{}]".format(x, y))
                image[y-1:y+1, x-1:x+1] = 0
            # elif range > msg.rang_max:
            #     x, y = self.getXY(msg.range_max, msg.angle_min,
            #                       msg.angle_increment, i)

        cv2.imwrite(self.root_save+"/IMAGE/"+image_name, image)

    def getXY(self, range, angle_min, angle_increment, index):
        angle = angle_min + index * angle_increment
        if angle > np.pi:
            angle = angle - 2*np.pi
        elif angle < -np.pi:
            angle = angle + 2*np.pi
        x = range * np.cos(angle)
        y = range * np.sin(angle)

        return x, y

    def getRangeTheta(self, msg):
        range_theta = []
        for i, rang in enumerate(msg.ranges):

            if rang > msg.range_max:
                rang = 0
            range_theta.append(rang)
            angle = msg.angle_min + i * msg.angle_increment
            if angle > np.pi:
                angle = angle - 2*np.pi
            elif angle < -np.pi:
                angle = angle + 2*np.pi
            range_theta.append(angle)
        return range_theta


def main(args):
    Bag_TO_CSV(
        bag_file_dir="/home/anis/catkin_ws/src/dataset_filter/data/bagfiles ", root_save="/home/anis/catkin_ws/src/dataset_filter/data/extracted_data")
    rospy.init_node('bagToCsvParser', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)
