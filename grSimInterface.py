import sys
import socket
import struct
import time
from threading import Thread
from threading import Lock

sys.path.append("proto/build")

import grSim_Commands_pb2
import grSim_Packet_pb2
import grSim_Replacement_pb2
import messages_robocup_ssl_detection_pb2
import messages_robocup_ssl_geometry_pb2
import messages_robocup_ssl_refbox_log_pb2
import messages_robocup_ssl_wrapper_pb2


class RealTimeGrSim:
    def run_rx(self):
        #multicast_group = '224.5.23.2'
        server_address = ('', 10020)

        # Create the socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to the server address
        sock.bind(server_address)

        while not self.stop_flag:
            data, addr = sock.recvfrom(1024)
            wrapper = messages_robocup_ssl_wrapper_pb2.SSL_WrapperPacket()

            # unpack
            with self.lock:
                me_x, me_y, ball_x, ball_y = self.state

                try:
                    wrapper.ParseFromString(data)

                    for robot in wrapper.detection.robots_blue:
                        if (robot.robot_id == 1):
                            me_x = robot.x
                            me_y = robot.y

                    for ball in wrapper.detection.balls:
                        ball_x = ball.x
                        ball_y = ball.y

                except:
                    pass

                # repack
                self.state = (me_x, me_y, ball_x, ball_y)

            time.sleep(0.001)

    def run_tx(self):
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_address = ('', 20011)

        while not self.stop_flag:
            packet = grSim_Packet_pb2.grSim_Packet()

            with self.lock:
                if self.reset_env:
                    self.reset_env = False

                    replace_packet = packet.replacement

                    ball_replace = replace_packet.ball
                    
                    ball_replace.pos.x = 0.0
                    ball_replace.pos.y = 0.0
                    ball_replace.vel.x = 0.0
                    ball_replace.vel.y = 0.0

                    robot_replace = replace_packet.robots.add()
                    robot_replace.x = -4.2
                    robot_replace.y = 0.0
                    robot_replace.dir = 0.0
                    robot_replace.id = 1
                    robot_replace.yellowteam = False

            commands_packet = packet.commands
            commands_packet.timestamp = time.time()
            commands_packet.isteamyellow = 0

            commands = packet.commands.robot_commands
            c = commands.add()

            vel_tan, vel_norm, vel_ang = self.vel_action

            c.id = 1
            c.kickspeedx = 0
            c.kickspeedz = 0
            c.veltangent = vel_tan
            c.velnormal = vel_norm
            c.velangular = vel_ang
            c.spinner = False
            c.wheelsspeed = False

            send_sock.sendto(packet.SerializeToString(), send_address)

            time.sleep(0.1)

    def __init__(self):
        self.lock = Lock()

        # (me_x, me_y, ball_x, ball_y)
        self.state = (0.0, 0.0, 0.0, 0.0)
        # (tan, norm, ang)
        self.vel_action = (0.0, 0.0, 0.0)

        self.stop_flag = False
        self.reset_env = False

        self.rx_thread = Thread(target = self.run_rx)
        self.tx_thread = Thread(target = self.run_tx)

        self.rx_thread.start()
        self.tx_thread.start()

    def stop(self):
        self.stop_flag = True

    def is_done(self):
        return time.time() - self.start_time > 5

    def sample_reward(self):
        # return some function robot and ball position
        me_x, me_y, ball_x, ball_y = self.state
        dst = pow(pow((me_x - ball_x) / 1000, 2) + pow((me_y - ball_y) / 1000, 2), 0.5)
        return 1.0 if dst < 1.0 else 0.0

    def set_current_action(self, vel_tan, vel_norm, vel_ang):
        with self.lock:
            self.vel_action = (vel_tan, vel_norm, vel_ang)

    def get_current_state(self):
        return self.state

    def start(self):
        self.start_time = time.time()

    def reset(self):
        with self.lock:
            self.reset_env = True
            self.vel_action = (0, 0, 0)

        while self.reset_env:
            time.sleep(0.1)

if __name__ == "__main__":
    env = RealTimeGrSim()

    time.sleep(0.5)

    try:
        while True:
            env.reset() 
            env.start()
            while not env.is_done():
                print(env.sample_reward())
                env.set_current_action(1, 0, 0.01)
                time.sleep(0.05)

            env.reset() 
    except KeyboardInterrupt:
        env.stop()

