import sys
import socket
import struct

sys.path.append("proto/build")

import grSim_Commands_pb2
import grSim_Packet_pb2
import grSim_Replacement_pb2
import messages_robocup_ssl_detection_pb2
import messages_robocup_ssl_geometry_pb2
import messages_robocup_ssl_refbox_log_pb2
import messages_robocup_ssl_wrapper_pb2

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def main():

    #multicast_group = '224.5.23.2'
    server_address = ('', 10020)

    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    sock.bind(server_address)


    while True:
        data, addr = sock.recvfrom(1024)
        wrapper = messages_robocup_ssl_wrapper_pb2.SSL_WrapperPacket()
        try:
            wrapper.ParseFromString(data)
        except:
            pass
        for robot in wrapper.detection.robots_blue:
            if (robot.robot_id == 1):
                print(robot.x)
                print(robot.y)

    

if __name__ == "__main__":
    main()
