import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TrafficConeDetectionNode(Node):
    def __init__(self):
        super().__init__('traffic_cone_detection_node')
        self.subscription = self.create_subscription(
            Image, '/zed/zed_node/rgb/image_rect_color', self.image_callback, 10)
        self.publisher = self.create_publisher(
            Image, '/traffic_cone_detection/image', 10)
        self.bridge = CvBridge()
        self.paused = False
        
        
        cv2.namedWindow('Trackbars')
        cv2.createTrackbar('Lower H Orange', 'Trackbars', 0, 179, self.nothing)
        cv2.createTrackbar('Lower S Orange', 'Trackbars', 104, 255, self.nothing)
        cv2.createTrackbar('Lower V Orange', 'Trackbars', 140, 255, self.nothing)
        cv2.createTrackbar('Upper H Orange', 'Trackbars', 4, 179, self.nothing)
        cv2.createTrackbar('Upper S Orange', 'Trackbars', 255, 255, self.nothing)
        cv2.createTrackbar('Upper V Orange', 'Trackbars', 255, 255, self.nothing)

        
        cv2.createTrackbar('Lower H White', 'Trackbars', 0, 179, self.nothing)
        cv2.createTrackbar('Lower S White', 'Trackbars', 220, 255, self.nothing)
        cv2.createTrackbar('Lower V White', 'Trackbars', 141, 255, self.nothing)
        cv2.createTrackbar('Upper H White', 'Trackbars', 179, 179, self.nothing)
        cv2.createTrackbar('Upper S White', 'Trackbars', 255, 255, self.nothing)
        cv2.createTrackbar('Upper V White', 'Trackbars', 255, 255, self.nothing)

    def nothing(self, x):
        pass

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        
        lower_h_orange = cv2.getTrackbarPos('Lower H Orange', 'Trackbars')
        lower_s_orange = cv2.getTrackbarPos('Lower S Orange', 'Trackbars')
        lower_v_orange = cv2.getTrackbarPos('Lower V Orange', 'Trackbars')
        upper_h_orange = cv2.getTrackbarPos('Upper H Orange', 'Trackbars')
        upper_s_orange = cv2.getTrackbarPos('Upper S Orange', 'Trackbars')
        upper_v_orange = cv2.getTrackbarPos('Upper V Orange', 'Trackbars')
        
        
        lower_h_white = cv2.getTrackbarPos('Lower H White', 'Trackbars')
        lower_s_white = cv2.getTrackbarPos('Lower S White', 'Trackbars')
        lower_v_white = cv2.getTrackbarPos('Lower V White', 'Trackbars')
        upper_h_white = cv2.getTrackbarPos('Upper H White', 'Trackbars')
        upper_s_white = cv2.getTrackbarPos('Upper S White', 'Trackbars')
        upper_v_white = cv2.getTrackbarPos('Upper V White', 'Trackbars')
        
        
        lower_orange = np.array([lower_h_orange, lower_s_orange, lower_v_orange])
        upper_orange = np.array([upper_h_orange, upper_s_orange, upper_v_orange])
        lower_white = np.array([lower_h_white, lower_s_white, lower_v_white])
        upper_white = np.array([upper_h_white, upper_s_white, upper_v_white])

        
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        
        mask_orange = cv2.inRange(hsv_image, lower_orange, upper_orange)
        
       
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

        
        mask_combined = cv2.bitwise_or(mask_orange, mask_white)

        
        mask_combined = cv2.GaussianBlur(mask_combined, (5, 5), 0)

        
        kernel = np.ones((5, 5), np.uint8)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

        
        cv_image_masked = cv2.bitwise_and(cv_image, cv_image, mask=mask_combined)
        cv2.imshow("Combined Mask", mask_combined)
        cv2.imshow("Masked Image", cv_image_masked)

        
        gray_image = cv2.cvtColor(cv_image_masked, cv2.COLOR_BGR2GRAY)

        
        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        min_width = 30
        min_height = 30

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width and h >= min_height:
                cv_image = cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        result_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        self.publisher.publish(result_msg)

       
        key = cv2.waitKey(1)
        if key == ord('p'):  
            self.paused = not self.paused
        while self.paused:
            key = cv2.waitKey(1)
            if key == ord('p'): 
                self.paused = not self.paused

def main(args=None):
    rclpy.init(args=args)
    print("Adjust the sliders as needed")
    print("Press p to pause")
    node = TrafficConeDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

