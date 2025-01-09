import rclpy, math, numpy, yaml
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action.client import GoalStatus
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy
from decimal import Decimal

class MoveTocertainArea(Node):
    def __init__(self):
        super().__init__('mapping')

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

        # /map 토픽 구독 (맵 데이터)
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # /odom 토픽 구독 (로봇의 위치 정보)
        self.pose_subscription = self.create_subscription(
            Odometry,
            '/odom',  # 로봇의 위치를 받는 토픽 (Odometry 사용)
            self.pose_callback,
            qos_profile = qos_profile
        )

        # navigate_to_pose 액션 클라이언트
        self.client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        self.get_logger().info("실행 확인")

        # 확실한 지역 좌표 리스트
        self.certain_areas = []
        
        # 방문한 지점
        self.visit_areas = []
        
        # 로봇 위치 초기화
        self.robot_position = [0,0]

        # 벽 주변 픽셀
        self.wall_near_pixel = []

        # 맵 데이터
        self.map_data = None
        self.map_metadata = None

        # 3초에 한번씩 목표 업데이트
        self.timer = self.create_timer(1.0, self.timer_callback)


    def map_callback(self, msg: OccupancyGrid):
        # 맵 데이터 저장
        self.map_data = msg
        self.map_metadata = msg.info
        
        # 맵 데이터 갈 곳 정리
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # 확실한 지역 찾기
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                # 0 확실히 아는 곳
                if msg.data[idx] == 0:
                    x = float(Decimal(origin_x + j * resolution).quantize(Decimal('0.1')))
                    y = float(Decimal(origin_y + i * resolution).quantize(Decimal('0.1')))

                    if [x,y] not in self.certain_areas:
                        self.certain_areas.append([x,y])

                if msg.data[idx] == 100:
                    wall_x = float(Decimal(origin_x + j * resolution).quantize(Decimal('0.1')))
                    wall_y = float(Decimal(origin_y + i * resolution).quantize(Decimal('0.1')))

                    for dx in numpy.arange(-0.3, 0.4, 0.1):
                        for dy in numpy.arange(-0.3, 0.4, 0.1):
                            wall_pixel = [float(Decimal(wall_x + dx).quantize(Decimal('0.1'))), float(Decimal(wall_y + dy).quantize(Decimal('0.1')))]
                            if wall_pixel not in self.wall_near_pixel:
                                self.wall_near_pixel.append(wall_pixel)


    def pose_callback(self, msg: Odometry):
        # 로봇의 현재 위치를 받아옴 (맵 좌표계에서의 x, y 위치)
        self.robot_position = [float(Decimal(msg.pose.pose.position.x).quantize(Decimal('0.1'))), float(Decimal(msg.pose.pose.position.y).quantize(Decimal('0.1')))]

        # 목표 지점 범위 내의 지점 추가
        self.add_visited_area(self.robot_position)


    def timer_callback(self):
        # 가장 멀리 있는 지역 찾기
        self.move_to_certain_area(self.get_farthest_area())

        self.timer.cancel()


    def move_to_certain_area(self, target):
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
        self.get_logger().info(f"목표 지점 이동: {target}")
        self.send_goal_future = self.client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.navigate_to_pose_action_goal)


    def navigate_to_pose_action_goal(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warning('목표 지점 설정 거부')
            return
        
        action_result_future = goal_handle.get_result_async()
        action_result_future.add_done_callback(self.navigate_to_pose_action_result)


    def navigate_to_pose_action_result(self, future):
        action_status = future.result().status
        #action_result = future.result().result
        
        if action_status == GoalStatus.STATUS_SUCCEEDED:
            # 가장 먼 지점 찾기
            self.get_logger().info('다음 목표 지점 설정')

            temp = self.get_farthest_area()
            if temp is not None:
                self.move_to_certain_area(temp)
            elif temp is None:
                self.get_logger().info("맵 구성 완료 맵 저장")
                self.save_map()
                
                temp = sorted(self.certain_areas, key=lambda area: area[0])
                # self.get_logger().info(f"{temp}")

                # 확실한 지점에서 벽 주변 제외
                result = [item for item in self.certain_areas if item not in self.wall_near_pixel]           
                
                # 기존 내용 지우고 새로 쓰기
                with open('/home/rokey/Desktop/sorted_areas.txt', 'w') as f:
                    for area in result:
                        f.write(f"{area[0]}, {area[1]}\n")

        else:
            self.get_logger().warning('목표 지점에 도달하지 못했습니다.') 


    def get_farthest_area(self):
        # 가장 멀리 있는 확실한 지역 찾기
        max_distance = -1
        farthest_area = None

        for i in range(len(self.certain_areas)):
            area = self.certain_areas[i]

            if area not in self.visit_areas:
                # 가장 멀리 있는 지역 찾기
                distance = math.sqrt((area[0] - self.robot_position[0]) ** 2 + (area[1] - self.robot_position[1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_area = area

        # self.visit_areas.append(farthest_area)

        if farthest_area is not None:
            return farthest_area
        
        elif farthest_area is None:
            return
            

    def add_visited_area(self, target):
        # 현재 위치의 범위에 해당하는 좌표들 추가        
        for dx in numpy.arange(-0.9, 0.9, 0.1):
            for dy in numpy.arange(-0.9, 0.9, 0.1):
                visited_area = [float(Decimal(target[0] + dx).quantize(Decimal('0.1'))), float(Decimal(target[1] + dy).quantize(Decimal('0.1')))]
                if visited_area not in self.visit_areas:
                    self.visit_areas.append(visited_area)
                    self.get_logger().info('로봇 방문 주변 저장')


    def save_map(self):
        if self.map_data is None:
            self.get_logger().warning("맵 데이터가 없습니다. 저장할 수 없습니다.")
            return

        # 맵 데이터를 NumPy 배열로 변환
        map_array = numpy.array(self.map_data.data, dtype=numpy.int8).reshape(
            (self.map_metadata.height, self.map_metadata.width)
        )

        # 빈 공간(0)은 255, 장애물(100)은 0으로 설정하여 바이너리 이미지로 변환
        # 0은 빈 공간, 100은 장애물, 255는 알 수 없는 공간을 0-255로 변환
        map_array[map_array == 0] = 254
        map_array[map_array == 100] = 0
        map_array[map_array == -1] = 205  # 알 수 없는 공간은 205로 설정

        # PGM 파일로 맵 저장
        map_pgm_path = '/home/rokey/Desktop/map.pgm'
        with open(map_pgm_path, 'wb') as f:
            # PGM 헤더 작성
            f.write(b'P5\n')
            f.write(f"{self.map_metadata.width} {self.map_metadata.height}\n".encode('utf-8'))
            f.write(b'255\n')
            # 바이너리 이미지 데이터 작성
            map_array.tofile(f)

        # YAML 파일로 메타데이터 저장
        map_yaml_path = '/home/rokey/Desktop/map.yaml'
        map_info = {
            'image': 'map.pgm',
            'resolution': self.map_metadata.resolution,
            'origin': [self.map_metadata.origin.position.x, self.map_metadata.origin.position.y, self.map_metadata.origin.orientation.z],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196
        }
        with open(map_yaml_path, 'w') as f:
            yaml.dump(map_info, f)

        self.get_logger().info(f"맵이 {map_pgm_path}와 {map_yaml_path}에 저장되었습니다.")


def main(args=None):
    rclpy.init(args=args)
    move_node = MoveTocertainArea()
    rclpy.spin(move_node)
    move_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  