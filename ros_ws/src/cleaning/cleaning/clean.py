import rclpy, math, numpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rclpy.action.client import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from decimal import Decimal
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import OccupancyGrid
# import numpy as np

class Move(Node):
    def __init__(self):
        super().__init__('move')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # 신뢰성: BEST_EFFORT
            history=QoSHistoryPolicy.KEEP_LAST,            # 히스토리 정책: KEEP_LAST
            depth=1,                                       # 큐 크기: 1
            durability=QoSDurabilityPolicy.VOLATILE,       # 지속성: VOLATILE
            lifespan=rclpy.duration.Duration(seconds=0),   # Lifespan: Infinite (기본값)
            deadline=rclpy.duration.Duration(seconds=0),   # Deadline: Infinite (기본값)
            liveliness=QoSLivelinessPolicy.AUTOMATIC,      # Liveliness: AUTOMATIC
            liveliness_lease_duration=rclpy.duration.Duration(seconds=0)  # Liveliness lease duration: Infinite
        )

        # navigate_to_pose 액션 클라이언트
        self.client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        # /odom 토픽 구독 (로봇의 위치 정보)
        self.pose_subscription = self.create_subscription(
            Odometry,
            '/odom',  # 로봇의 위치를 받는 토픽 (Odometry 사용)
            self.pose_callback,
            qos_profile = qos_profile
        )

        # Marker publisher 생성
        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )


        #  # /scan 서브스크라이브
        # self.scan_subscription = self.create_subscription(
        #     LaserScan,
        #     '/scan',  # 라이더 데이터가 퍼블리시되는 토픽
        #     self.scan_callback,
        #     10  # 큐 사이즈
        # )

        # # 왼쪽 빈공간 거리
        # self.left_distance = 0.0
        # # 전진 빈공간 거리
        # self.center_distance = 0.0
        # # 오른쪽 빈공간 거리
        # self.right_distance = 0.0

        # 청소 할 지역
        self.certain_areas = []

        try:
            with open('/home/rokey/Desktop/sorted_areas.txt', 'r') as file:
                for line in file:
                    # 각 줄에서 공백을 제거하고, ','로 분리하여 x, y 값을 가져오기
                    coords = line.strip().split(',')
                    if len(coords) == 2:  # x와 y가 있을 경우에만 처리
                        try:
                            x = float(coords[0])  # x 값
                            y = float(coords[1])  # y 값
                            self.certain_areas.append([x, y])  # 리스트에 좌표 추가
                        except ValueError:
                            self.get_logger().warning(f"잘못된 좌표 형식: {line.strip()}")
        except FileNotFoundError:
            self.get_logger().warning("파일을 찾을 수 없습니다: /home/rokey/Desktop/sorted_areas.txt")

        # self.get_logger().info(f'{self.certain_areas}')

        # 청소 좌표
        self.target = None

        # 청소한 지역
        self.clean_areas = []

        # 로봇 위치 초기화
        self.robot_position = [0.0,0.0]

        # 청소 알고리즘
        self.timer = self.create_timer(3.0, self.timer_callback)
        self.timer2 = self.create_timer(0.3, self.timer2_callback)


    # def scan_callback(self, msg: LaserScan):
    #     self.left_distance = msg.ranges[int(len(msg.ranges)*(2/4))]
    #     self.center_distance = msg.ranges[int(len(msg.ranges)*(1/4))]
    #     self.right_distance = msg.ranges[int(len(msg.ranges))-1]

    #     # self.get_logger().info(f"{self.left_distance}")
    #     # self.get_logger().info(f"{self.center_distance}")
    #     # self.get_logger().info(f"{self.right_distance}")
    

    def timer_callback(self):
        # 가장 가까운 지역 찾기
        temp = self.get_close_area()
        self.move_to_dirty_area(temp)

        self.timer.cancel()

    def timer2_callback(self):
        marker = Marker()
        marker.header.frame_id = "map"  # 마커의 좌표계, 일반적으로 "map" 또는 "odom"
        marker.type = Marker.CYLINDER  # 원기둥
        marker.action = Marker.ADD  # 마커 추가
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()  # 마커 지속 시간 (0은 계속 존재)

        # 마커 색상 설정
        marker.color = ColorRGBA()
        marker.color.r = 0.0  # 빨간색
        marker.color.g = 1.0  # 초록색
        marker.color.b = 0.0  # 파란색
        marker.color.a = 1.0  # 투명도 (0: 투명, 1: 불투명)

        # 마커 크기 (반지름과 높이를 설정)
        marker.scale.x = 0.05 * 2  # 원의 x 반지름 (2배로 설정하여 더 크게 만듬)
        marker.scale.y = 0.05 * 2  # 원의 y 반지름 (2배로 설정하여 더 크게 만듬)
        marker.scale.z = 0.01  # 매우 얇은 높이 (2D 원처럼 보이도록 설정)

        # 각 청소 지역에 대해 마커 추가
        for i, area in enumerate(self.clean_areas):
            marker.id = i  # 각 마커마다 고유한 id 부여
            marker.pose.position = Point(x=area[0], y=area[1], z=0.0)  # x, y 좌표 설정
            self.marker_pub.publish(marker)  # 마커를 퍼블리시

    def pose_callback(self, msg: Odometry):
        # 로봇의 현재 위치를 받아옴 (맵 좌표계에서의 x, y 위치)
        self.robot_position = [float(Decimal(msg.pose.pose.position.x).quantize(Decimal('0.1'))), float(Decimal(msg.pose.pose.position.y).quantize(Decimal('0.1')))]

        # 목표 지점 범위 내의 지점 추가
        self.add_visited_area(self.robot_position)


    def move_to_dirty_area(self, target):
        # navigate_to_pose 액션 클라이언트 사용
        if not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Action server not available!")
            return

        # 목표 위치 설정 (PoseStamped 사용)
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'map'  # 맵 좌표계 사용
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = target[0]
        target_pose.pose.position.y = target[1]
        target_pose.pose.orientation.w = 1.0  # 회전 각도 설정 (기본값으로 회전하지 않음)

        # 액션 목표 설정
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        # 액션 서버로 목표 전송
        self.target = target
        self.get_logger().info(f"청소 지점으로 이동: {target}")
        self.send_goal_future = self.client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.navigate_to_pose_action_goal)


    def navigate_to_pose_action_goal(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warning('목표지점 설정 거부')
            return
        
        action_result_future = goal_handle.get_result_async()
        action_result_future.add_done_callback(self.navigate_to_pose_action_result)


    def navigate_to_pose_action_result(self, future):
        action_status = future.result().status
        #action_result = future.result().result
        
        if action_status == GoalStatus.STATUS_SUCCEEDED:
            if self.target not in self.clean_areas:
                self.clean_areas.append(self.target)
                self.get_logger().info(f'남은 청소 지점 개수 :{len(self.certain_areas) - len(self.clean_areas)}')

            self.get_logger().info('다음 청소 지점 설정')

            temp = self.get_close_area()
            if temp is not None:
                self.move_to_dirty_area(temp)
            elif temp is None:
                self.get_logger().info("청소 완료")

        else:
            self.get_logger().warning('목표 지점에 도달하지 못했습니다.') 

    def get_close_area(self):
        # 가장 가까운 지역 찾기
        min_distance = 100
        closest_area = None

        for i in range(len(self.certain_areas)):
            area = self.certain_areas[i]

            if area not in self.clean_areas:
                # 가장 가까운 지역 찾기
                distance = math.sqrt((area[0] - self.robot_position[0]) ** 2 + (area[1] - self.robot_position[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_area = area

        if closest_area is not None:
            return closest_area
        
        elif closest_area is None:
            return


    def add_visited_area(self, target):
        # 현재 위치의 범위에 해당하는 좌표들 추가        
        for dx in numpy.arange(-0.1, 0.2, 0.1):
            for dy in numpy.arange(-0.1, 0.2, 0.1):
                visited_area = [float(Decimal(target[0] + dx).quantize(Decimal('0.1'))), float(Decimal(target[1] + dy).quantize(Decimal('0.1')))]
                if visited_area not in self.clean_areas:
                    self.clean_areas.append(visited_area)
                    self.get_logger().info(f'남은 청소 지점 개수 :{len(self.certain_areas) - len(self.clean_areas)}')


def main(args=None):
    rclpy.init(args=args)
    move_node = Move()

    rclpy.spin(move_node)

    move_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()