#!/usr/bin/env python
#encoding: utf8

# 上田本のフェイストラッキングを参考にクラスをまるごと書き換える
# カスケード分類器を作るのは失敗したので，色相からボールを抽出して重心を求める
# 隙間はモルフォロジーで埋める
import rospy, cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# class FaceToFace():
#     def __init__(self):
          # トピック名，トピックの型，コールバック関数の順
#         sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
#         self.bridge = CvBridge()
#         self.image_org = None
# 
#     def get_image(self, img):
#         try:
#             self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
#         except CvBridgeError as e:
#             rospy.logerr(e)
# 
#     def detect_face(self):
#         if self.image_org is None:
#             return None
          # イメージの読み込み
#         org = self.image_org
#         gimg = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
          # カスケード型分類器のための顔検出用データ
#         classifier = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
#         cascade = cv2.CascadeClassifier(classifier)
#         face = cascade.detectMultiScale(gimg, 1.1, 1, cv2.CASCADE_FIND_BIGGEST_OBJECT)
#         if len(face) == 0:
#             return None
# 
#         r = face[0]
          # rに入っているのは4要素のリスト(cv::Rectに相当)．.x,.y,.width,.height
          # つまりtuple(r[0:2]でcv::Pointにあたるものを渡している
          # tuple(r[0:2]+r[2:4]で(self.x+self.width, self.y+self.height)と同等
#         cv2.rectangle(org, tuple(r[0:2]), tuple(r[0:2]+r[2:4]), (0, 255, 255), 4)
#         cv2.imwrite("/tmp/image.jpg", org)

# 
#         return "detected"


class ObjectTracker():
    def __init__(self):
        sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
        self.bridge = CvBridge()
        self.image_org = None
        self.pub = rospy.Publisher("object", Image, queue_size=1)

    def get_image(self, img):
        try:
            # この時点でnparray型になっている
            self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def detect_orange(self):
        if self.image_org is None:
            return None
        org = self.image_org
        hsv = cv2.cvtColor(org, cv2.COLOR_BGR2HSV)

        min_hsv_orange = np.array([5, 80, 80])
        max_hsv_orange = np.array([15, 255, 255])
        binary = cv2.inRange(hsv, min_hsv_orange, max_hsv_orange)
        # 遅いが円形のカーネル
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # 速いが矩形のカーネル
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 3)
        center_img = self.detect_center(binary)
        
        self.monitor(center_img)
        #cv2.imwrite("/tmp/image.jpg", org)

    def detect_center(self, binary):
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        center_img = cv2.drawContours(binary, contours, -1, (0, 255, 0), 20)
        center_img = cv2.cvtColor(center_img, cv2.COLOR_GRAY2RGB)
        #cv2.imwrite("/tmp/image.jpg", center_img)
        return center_img

    def monitor(self, org):
        self.pub.publish(self.bridge.cv2_to_imgmsg(org, "rgb8"))


        return "detected"
if __name__ == '__main__':
    rospy.init_node('object_tracking')
    ot = ObjectTracker()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.loginfo(ot.detect_orange())
        rate.sleep()

