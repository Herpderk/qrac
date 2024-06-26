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
import threading
import logging
import numpy as np
import cflib
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from qrac.optitrack.NatNetClient import NatNetClient
import qrac.optitrack.DataDescriptions
import qrac.optitrack.MoCapData


class CrazyflieFeedback():
    def __init__(
        self,
        period_ms: int,
        uri="radio://0/80/2M/E7E7E7E7E7"
    ):
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers()
        
        logger = self._init_logger(period_ms=period_ms)
        self._thread = threading.Thread(
            target=self._run_logger, args=(logger, uri)
        )
        self._run_flag = threading.Event()
        self._verbose = False
        self._milli_vels = ()

    def start(self, verbose=False):
        self._verbose = verbose
        self._thread.start()

    def stop(self):
        self._run_flag.set()
        self._thread.join()

    def get_velocities(self):
        return 0.001*np.array(self._milli_vels)

    def _logger_cb(self, timestamp, data, logger):
        self._milli_vels = (
            data["stateEstimateZ.vx"],
            data["stateEstimateZ.vy"],
            data["stateEstimateZ.vz"],
            data["stateEstimateZ.rateRoll"],
            data["stateEstimateZ.ratePitch"],
            data["stateEstimateZ.rateYaw"],
        )
        if self._verbose:
            print(f"\nvel: {self._milli_vels[0:3]}")
            print(f"ang vel: {self._milli_vels[3:6]}\n")

    def _run_logger(self, logger, uri: str):
        print("Starting Crazyflie Logger...")
        with SyncCrazyflie(uri, cf=Crazyflie()) as scf:
            scf.cf.log.add_config(logger)
            logger.data_received_cb.add_callback(self._logger_cb)
            logger.start()
            self._run_flag.wait()
            logger.stop()

    def _init_logger(self, period_ms: int):
        logger = LogConfig(name="Vel_Estimator", period_in_ms=period_ms)
        logger.add_variable("stateEstimateZ.vx", "int16_t")
        logger.add_variable("stateEstimateZ.vy", "int16_t")
        logger.add_variable("stateEstimateZ.vz", "int16_t")
        logger.add_variable("stateEstimateZ.rateRoll", "int16_t")
        logger.add_variable("stateEstimateZ.ratePitch", "int16_t")
        logger.add_variable("stateEstimateZ.rateYaw", "int16_t")
        return logger


class OptiTrackFeedback():
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
        self._verbose = False
        self._pos = ()
        self._q = ()

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

    def start(self, verbose=False):
        self._verbose = verbose
        self._run_client()

    def stop(self):
        self._client.shutdown()

    def get_pose(self):
        return np.concatenate((
            self._pos,
            self._q
        ))

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def _rigid_body_cb(self, frame_id, pos, quat):
        if frame_id != self._id:
            pass
        else:
            self._pos = pos
            self._q = (quat[3], quat[0], quat[1], quat[2])
            if self._verbose:
                print( "Received frame for rigid body", frame_id )
                print(f"pos: {pos}")
                print(f"quat: {quat}\n")

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
        print("Starting NatNet client...")
        is_running = self._client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

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
