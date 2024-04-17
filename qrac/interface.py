﻿#Copyright © 2018 Naturalpoint
#
#Licensed under the Apache License, Version 2.0 (the "License")
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import sys
import time
import numpy as np
from qrac.optitrack.NatNetClient import NatNetClient
import qrac.optitrack.DataDescriptions
import qrac.optitrack.MoCapData


class OptiTrackInterface():
    def __init__(
        self,
        frame_id: int,
        server_addr="127.0.0.1",
        local_addr="127.0.0.1",
        multicast_addr="239.255.42.99",
        command_port=1510,
        data_port=1511,
        use_multicast=True,
    ):
        self._id = frame_id
        self._pos = np.zeros(3)
        self._q = np.zeros(4)

        self._client = NatNetClient(
            rigid_body_cb=self._rigid_body_cb,
            new_frame_cb=self._new_frame_cb,
            server_addr=server_addr,
            local_addr=local_addr,
            multicast_addr=multicast_addr,
            command_port=command_port,
            data_port=data_port,
            use_multicast=use_multicast
        )

    def start(self):
        self._run_client()

    def stop(self):
        self._client.shutdown()

    def get_position(self):
        return self._pos

    def get_quaternion(self):
        return self._q

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def _rigid_body_cb(self, frame_id, pos, quat):
        if frame_id != self._id:
            pass
        else:
            self._pos[:] = pos
            self._q[:] = quat
            print( "Received frame for rigid body", frame_id )
            print( "Received frame for rigid body", frame_id," ",pos," ",quat)

    # This is a callback function that gets connected to the NatNet client
    # and called once per mocap frame.
    def _new_frame_cb(self, data_dict):
        order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                    "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
        dump_args = False
        if dump_args == True:
            out_string = "    "
            for key in data_dict:
                out_string += key + "="
                if key in data_dict :
                    out_string += data_dict[key] + " "
                out_string+="/"
            print(out_string)

    def _run_client(self):
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = self._client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        #is_looping = True
        is_looping = False
        time.sleep(1)
        if self._client.connected() is False:
            print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        print_configuration(self._client)
        print("\n")
        print_commands(self._client.can_change_bitstream_version())


        while is_looping:
            inchars = input('Enter command or (\'h\' for list of commands)\n')
            if len(inchars)>0:
                c1 = inchars[0].lower()
                if c1 == 'h' :
                    print_commands(self._client.can_change_bitstream_version())
                elif c1 == 'c' :
                    print_configuration(self._client)
                elif c1 == 's':
                    request_data_descriptions(self._client)
                    time.sleep(1)
                elif (c1 == '3') or (c1 == '4'):
                    if self._client.can_change_bitstream_version():
                        tmp_major = 4
                        tmp_minor = 1
                        if(c1 == '3'):
                            tmp_major = 3
                            tmp_minor = 1
                        return_code = self._client.set_nat_net_version(tmp_major,tmp_minor)
                        time.sleep(1)
                        if return_code == -1:
                            print("Could not change bitstream version to %d.%d"%(tmp_major,tmp_minor))
                        else:
                            print("Bitstream version at %d.%d"%(tmp_major,tmp_minor))
                    else:
                        print("Can only change bitstream in Unicast Mode")

                elif c1 == 'p':
                    sz_command="TimelineStop"
                    return_code = self._client.send_command(sz_command)
                    time.sleep(1)
                    print("Command: %s - return_code: %d"% (sz_command, return_code) )
                elif c1 == 'r':
                    sz_command="TimelinePlay"
                    return_code = self._client.send_command(sz_command)
                    print("Command: %s - return_code: %d"% (sz_command, return_code) )
                elif c1 == 'o':
                    tmpCommands=["TimelinePlay",
                                "TimelineStop",
                                "SetPlaybackStartFrame,0",
                                "SetPlaybackStopFrame,1000000",
                                "SetPlaybackLooping,0",
                                "SetPlaybackCurrentFrame,0",
                                "TimelineStop"]
                    for sz_command in tmpCommands:
                        return_code = self._client.send_command(sz_command)
                        print("Command: %s - return_code: %d"% (sz_command, return_code) )
                    time.sleep(1)
                elif c1 == 'w':
                    tmp_commands=["TimelinePlay",
                                "TimelineStop",
                                "SetPlaybackStartFrame,1",
                                "SetPlaybackStopFrame,1500",
                                "SetPlaybackLooping,0",
                                "SetPlaybackCurrentFrame,100",
                                "TimelineStop"]
                    for sz_command in tmp_commands:
                        return_code = self._client.send_command(sz_command)
                        print("Command: %s - return_code: %d"% (sz_command, return_code) )
                    time.sleep(1)
                elif c1 == 't':
                    test_classes()

                elif c1 == 'j':
                    self._client.set_print_level(0)
                    print("Showing only received frame numbers and supressing data descriptions")
                elif c1 == 'k':
                    self._client.set_print_level(1)
                    print("Showing every received frame")

                elif c1 == 'l':
                    print_level = self._client.set_print_level(20)
                    print_level_mod = print_level % 100
                    if(print_level == 0):
                        print("Showing only received frame numbers and supressing data descriptions")
                    elif (print_level == 1):
                        print("Showing every frame")
                    elif (print_level_mod == 1):
                        print("Showing every %dst frame"%print_level)
                    elif (print_level_mod == 2):
                        print("Showing every %dnd frame"%print_level)
                    elif (print_level == 3):
                        print("Showing every %drd frame"%print_level)
                    else:
                        print("Showing every %dth frame"%print_level)

                elif c1 == 'q':
                    is_looping = False
                    self._client.shutdown()
                    break
                else:
                    print("Error: Command %s not recognized"%c1)
                print("Ready...\n")
        print("exiting")
   

def add_lists(totals, totals_tmp):
    totals[0]+=totals_tmp[0]
    totals[1]+=totals_tmp[1]
    totals[2]+=totals_tmp[2]
    return totals

def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s"% natnet_client.local_ip_address)
    print("  Server:          %s"% natnet_client.server_ip_address)
    print("  Command Port:    %d"% natnet_client.command_port)
    print("  Data Port:       %d"% natnet_client.data_port)

    changeBitstreamString = "  Can Change Bitstream Version = "
    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s"% natnet_client.multicast_address)
        changeBitstreamString+="false"
    else:
        print("  Using Unicast")
        changeBitstreamString+="true"

    #NatNet Server Info
    application_name = natnet_client.get_application_name()
    nat_net_requested_version = natnet_client.get_nat_net_requested_version()
    nat_net_version_server = natnet_client.get_nat_net_version_server()
    server_version = natnet_client.get_server_version()

    print("  NatNet Server Info")
    print("    Application Name %s" %(application_name))
    print("    MotiveVersion  %d %d %d %d"% (server_version[0], server_version[1], server_version[2], server_version[3]))
    print("    NatNetVersion  %d %d %d %d"% (nat_net_version_server[0], nat_net_version_server[1], nat_net_version_server[2], nat_net_version_server[3]))
    print("  NatNet Bitstream Requested")
    print("    NatNetVersion  %d %d %d %d"% (nat_net_requested_version[0], nat_net_requested_version[1],\
       nat_net_requested_version[2], nat_net_requested_version[3]))

    print(changeBitstreamString)
    #print("command_socket = %s"%(str(natnet_client.command_socket)))
    #print("data_socket    = %s"%(str(natnet_client.data_socket)))
    print("  PythonVersion    %s"%(sys.version))


def print_commands(can_change_bitstream):
    outstring = "Commands:\n"
    outstring += "Return Data from Motive\n"
    outstring += "  s  send data descriptions\n"
    outstring += "  r  resume/start frame playback\n"
    outstring += "  p  pause frame playback\n"
    outstring += "     pause may require several seconds\n"
    outstring += "     depending on the frame data size\n"
    outstring += "Change Working Range\n"
    outstring += "  o  reset Working Range to: start/current/end frame 0/0/end of take\n"
    outstring += "  w  set Working Range to: start/current/end frame 1/100/1500\n"
    outstring += "Return Data Display Modes\n"
    outstring += "  j  print_level = 0 supress data description and mocap frame data\n"
    outstring += "  k  print_level = 1 show data description and mocap frame data\n"
    outstring += "  l  print_level = 20 show data description and every 20th mocap frame data\n"
    outstring += "Change NatNet data stream version (Unicast only)\n"
    outstring += "  3  Request NatNet 3.1 data stream (Unicast only)\n"
    outstring += "  4  Request NatNet 4.1 data stream (Unicast only)\n"
    outstring += "General\n"
    outstring += "  t  data structures self test (no motive/server interaction)\n"
    outstring += "  c  print configuration\n"
    outstring += "  h  print commands\n"
    outstring += "  q  quit\n"
    outstring += "\n"
    outstring += "NOTE: Motive frame playback will respond differently in\n"
    outstring += "       Endpoint, Loop, and Bounce playback modes.\n"
    outstring += "\n"
    outstring += "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n"
    outstring += "         PacketClient \"192.168.10.14\" \"192.168.10.14\" Multicast\n"
    outstring += "         PacketClient \"127.0.0.1\" \"127.0.0.1\" u\n"
    outstring += "\n"
    print(outstring)

def request_data_descriptions(s_client):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF,    "",  (s_client.server_ip_address, s_client.command_port) )

def test_classes():
    totals = [0,0,0]
    print("Test Data Description Classes")
    totals_tmp = qrac.optitrack.DataDescriptions.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("Test MoCap Frame Classes")
    totals_tmp = qrac.optitrack.MoCapData.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("All Tests totals")
    print("--------------------")
    print("[PASS] Count = %3.1d"%totals[0])
    print("[FAIL] Count = %3.1d"%totals[1])
    print("[SKIP] Count = %3.1d"%totals[2])

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


if __name__ == "__main__":

    pass
