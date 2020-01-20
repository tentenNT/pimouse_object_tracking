#!/usr/bin/env python
#encoding: utf8

# 上田本のフェイストラッキングを参考にクラスをまるごと書き換える
# カスケード分類器を作るのは失敗したので，色相からボールを抽出して重心を求める
# 隙間はモルフォロジーで埋める

#todo
# 重心の方を向く
# 凸包
# ノイズ除去
# ボールの大きさによって前進後退


import rospy, cv2, math
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

LOWER_PERCENT = 0.01

class ObjectTracker():
    def __init__(self):
        sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
        self.bridge = CvBridge()
        self.image_org = None
        self.area_max = 0
        self.area_default = 0
        self.disp_default_now = 0
        self.area_whole = None
        self.pub = rospy.Publisher("object", Image, queue_size=1)
        self.hough = rospy.Publisher("hough", Image, queue_size=1)
        self.cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.wait_for_service("/motor_on")
        rospy.wait_for_service("/motor_off")
        rospy.on_shutdown(rospy.ServiceProxy("/motor_off", Trigger).call)
        rospy.ServiceProxy("/motor_on", Trigger).call()

    # どうもハフ変換は重いし上手く検出出来ない
    def test_hough(self, img):
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img,5)
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        self.hough.publish(self.bridge.cv2_to_imgmsg(img, "mono8"))

    def get_image(self, img):
        try:
            # この時点でnparray型になっている
            self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
            # ついでに面積計算
            self.area_whole = self.image_org.shape[0] * self.image_org.shape[1]
        except CvBridgeError as e:
            rospy.logerr(e)

    def detect_orange(self):
        if self.image_org is None:
            return None
        org = self.image_org

        hsv = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)

        # HSV色空間を用いてオレンジ色を抽出
        min_hsv_orange = np.array([15, 150, 40])
        max_hsv_orange = np.array([20, 255, 255])
        binary = cv2.inRange(hsv, min_hsv_orange, max_hsv_orange)
        # 遅いが円形のカーネル
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # 速いが矩形のカーネル
        # 不要かもしれない
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 2)
        center_img, point_cog = self.detect_center(binary)
        self.monitor(center_img)
        return point_cog

    # 2値画像の輪郭からモーメントを求める
    def detect_center(self, binary):
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.area_max = 0
        self.area_max_num = 0
        point_cog = (self.image_org.shape[1], self.image_org.shape[0])
        # 最大面積のインデックスを求める
        for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)

                if(self.area_max < area):
                    self.area_max = area
                    self.area_max_num = i

        if(self.area_default == 0 and self.area_max != 0):
            self.area_default = self.area_max
        self.disp_default_now = (self.area_default - self.area_max) / self.area_whole
        #print("self.area_max_percentage: {}".format(self.area_max/self.area_whole))
        #print("disp_default_now: {}".format(self.disp_default_now))

        center_img = cv2.drawContours(self.image_org, contours, self.area_max_num, (0, 255, 0), 5)

        if(self.area_max/self.area_whole > LOWER_PERCENT):
            self.area_old = self.area_max
            M = cv2.moments(contours[self.area_max_num])
            cog_x = int(M['m10'] / M['m00'])
            cog_y = int(M['m01'] / M['m00'])
            point_cog = (cog_x, cog_y)
            center_img = cv2.circle(center_img, point_cog, 15, (255, 0, 0), thickness=-1) 
        return center_img, point_cog

    def monitor(self, org):
        self.pub.publish(self.bridge.cv2_to_imgmsg(org, "bgr8"))
        return "detected"

    def rot_vel(self):
        point_cog = self.detect_orange()
        if (self.area_max/self.area_whole < LOWER_PERCENT):
            return 0.0
        wid = self.image_org.shape[1]/2
        pos_x_rate = (point_cog[0] - wid)*1.0/wid
        rot = -0.25*pos_x_rate*math.pi
        rospy.loginfo("detect %f", rot)
        return rot

    def control(self):
        m = Twist()
        if(self.area_max/self.area_whole > LOWER_PERCENT):
            if(self.disp_default_now > 0.01):
                m.linear.x = 0.1
                print("adv")
            elif(self.disp_default_now < -0.01):
                m.linear.x = -0.1
                print("back")
            else:
                m.linear.x = 0
                print("stay")
        m.angular.z = self.rot_vel()
        self.cmd_vel.publish(m)


if __name__ == '__main__':
    rospy.init_node('object_tracking')
    ot = ObjectTracker()

    rate = rospy.Rate(10)
    rate.sleep()
    while not rospy.is_shutdown():
        ot.control()
        rate.sleep()

