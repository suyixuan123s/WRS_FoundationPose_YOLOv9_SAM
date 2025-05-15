"""
Interface of robotic control over ethernet. Built for the GoFa 5 robot arm.
Author: Weiwei Wan, 20191024, osaka
"""

from multiprocessing import Process, Queue
from queue import Empty
import logging
import socket
import sys
from time import sleep
from collections import namedtuple
import numpy as np
from .autolab_core import RigidTransform
from .gofa_constants import GoFaConstants as GFC
from .gofa_state import GoFaState
from .gofa_motion_logger import GoFaMotionLogger
from .util import message_to_state, message_to_pose, message_to_torques
from .gofa_exceptions import GoFaCommException, GoFaControlException
import pickle
import struct

_RAW_RES = namedtuple('_RAW_RES', 'mirror_code res_code message')
_RES = namedtuple('_RES', 'raw_res data')
_REQ_PACKET = namedtuple('_REQ_PACKET', 'req timeout return_res')

METERS_TO_MM = 1000.0
MM_TO_METERS = 1.0 / METERS_TO_MM


class _GoFaEthernet(Process):

    def __init__(self, req_q, res_q, ip, port, bufsize, timeout, debug):
        Process.__init__(self)

        self._ip = ip
        self._port = port
        self._timeout = timeout
        self._bufsize = bufsize
        self._socket = None

        self._req_q = req_q
        self._res_q = res_q

        self._current_state = None

        self._debug = debug

    def run(self):
        logging.getLogger().setLevel(GFC.LOGGING_LEVEL)

        if self._debug:
            logging.info("In DEBUG mode. Messages will NOT be sent over socket.")
        else:
            self._reset_socket()

        try:
            while True:
                req_packet = self._req_q.get()
                if req_packet == "stop":
                    break
                res = self._send_request(req_packet)
                if req_packet.return_res:
                    self._res_q.put(res)
                sleep(GFC.PROCESS_SLEEP_TIME)

        except KeyboardInterrupt:
            self._stop()
            sys.exit(0)

        self._stop()

    def _stop(self):
        logging.info("Shutting down gofa ethernet interface")
        if not self._debug:
            self._socket.close()

    def _reset_socket(self):
        logging.debug('Opening socket on {0}:{1}'.format(self._ip, self._port))
        if self._socket != None:
            self._socket.close()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._ip, self._port))
        logging.debug('Socket successfully opened!')

    def _send_request(self, req_packet):
        logging.debug("Sending: {0}".format(req_packet))
        raw_res = None

        if self._debug:
            raw_res = '-1 1 MOCK RES for {0}'.format(req_packet)
        else:
            self._socket.settimeout(req_packet.timeout)

            while True:
                try:
                    self._socket.send(req_packet.req)
                    break
                except socket.error as e:
                    # TODO: better way to handle this mysterious bad file descriptor error
                    if e.errno == 9:
                        self._reset_socket()
            try:
                raw_res = self._socket.recv(self._bufsize).decode()
            except socket.error as e:
                if e.errno == 114:  # request time out
                    raise GoFaCommException('Request timed out: {0}'.format(req_packet))

        logging.debug("Received: {0}".format(raw_res))

        if raw_res is None or len(raw_res) == 0:
            raise GoFaCommException('Empty response! For req: {0}'.format(req_packet))

        tokens = raw_res.split()
        res = _RAW_RES(int(tokens[0]), int(tokens[1]), ' '.join(tokens[2:]))
        return res


class GoFaArm:
    """ Interface to a single arm of an ABB GoFa robot_s.
    Communicates with the robot_s over Ethernet.
    """

    def __init__(self, ip=GFC.IP, port=GFC.PORTS["server"], bufsize=GFC.BUFSIZE,
                 motion_timeout=GFC.MOTION_TIMEOUT, comm_timeout=GFC.COMM_TIMEOUT, process_timeout=GFC.PROCESS_TIMEOUT,
                 from_frame='tool', to_frame='base',
                 debug=GFC.DEBUG,
                 log_pose_histories=False, log_state_histories=False, ):
        '''Initializes a GoFaArm interface. This interface will communicate with one arm (port) on the GoFa Robot.
        This uses a subprocess to handle non-blocking socket communication with the RAPID server.

        Parameters
        ----------
            name : string
                    Name of the arm {'left', 'right'}
            ip : string formated ip address, optional
                    IP of GoFa Robot.
                    Default uses the one in GoFaConstants
            port : int, optional
                    Port of target arm's server.
                    Default uses the port for the left arm from GoFaConstants.
            bufsize : int, optional
                    Buffer size for ethernet responses
            motion_timeout : float, optional
                    Timeout for motion commands.
                    Default from GoFaConstants.MOTION_TIMEOUT
            comm_timeout : float, optional
                    Timeout for non-motion ethernet communication.
                    Default from GoFaConstants.COMM_TIMEOUT
            process_timeout : float, optional
                    Timeout for ethernet process communication.
                    Default from GoFaConstants.PROCESS_TIMEOUT
            from_frame : string, optional
                    String name of robot_s arm frame.
                    Default to "tool"
            to_frame : string, optional
                    String name of reference for robot_s frame
                    Default to "base"
            debug : bool, optional
                    Boolean to indicate whether or not in debug mode. If in debug mode no ethernet communication is attempted. Mock responses will be returned.
                    Default to GoFaConstants.DEBUG
            log_pose_histories : bool, optional
                    If True, uses gofa_history_logger to log pose histories. Enables usage of flush_pose_histories.
                    Defaults to False
            log_state_histories : bool, optional
                    If True, uses gofa_history_logger to log state histories. Enables usage of flush_state_histories.
                    Defaults to False
        '''
        self._motion_timeout = motion_timeout
        self._comm_timeout = comm_timeout
        self._process_timeout = process_timeout
        self._ip = ip
        self._port = port
        self._bufsize = bufsize
        self._from_frame = from_frame
        self._to_frame = to_frame
        self._debug = debug

        if log_pose_histories:
            self._pose_logger = GoFaMotionLogger()
        if log_state_histories:
            self._state_logger = GoFaMotionLogger()

        self.start()

        self._last_sets = {
            'zone': None,
            'speed': None,
            'tool': None,
        }

    def reset_settings(self):
        '''Reset zone, tool, and speed settings to their last known values. This is used when reconnecting to the RAPID server after a server restart.
        '''
        # set robot_s settings
        for key, val in self._last_sets.items():
            if val is not None:
                getattr(self, 'set_{0}'.format(key))(val)

    def reset(self):
        '''Resets the underlying gofa ethernet process and socket, and resets all the settings.
        '''
        # empty motion logs
        if hasattr(self, '_state_logger'):
            self._state_logger.reset_log()
        if hasattr(self, '_pose_logger'):
            self._pose_logger.reset_log()

        # terminate ethernet
        try:
            self._gofa_ethernet.terminate()
        except Exception:
            pass

        # start ethernet comm
        self._create_gofa_ethernet()

        self.reset_settings()

    def flush_pose_histories(self, filename):
        '''
        Parameters
        ----------
        filename : string
                Saves the pose history logger data to filename. Empties logger.
        '''
        self._pose_logger.flush_to_file(filename)

    def flush_state_histories(self, filename):
        '''
        Parameters
        ----------
        filename : string
                Saves the state history logger data to filename. Empties logger
        '''
        self._state_logger.flush_to_file(filename)

    def _create_gofa_ethernet(self):
        self._req_q = Queue()
        self._res_q = Queue()

        self._gofa_ethernet = _GoFaEthernet(self._req_q, self._res_q, self._ip, self._port,
                                            self._bufsize, self._comm_timeout, self._debug)
        self._gofa_ethernet.start()

    def start(self):
        '''Starts subprocess for ethernet communication.
        '''
        self._create_gofa_ethernet()

    def stop(self):
        '''Stops subprocess for ethernet communication. Allows program to exit gracefully.
        '''
        self._req_q.put("stop")
        try:
            self._gofa_ethernet.terminate()
        except Exception:
            pass

    def __del__(self):
        self.stop()

    def _request(self, req, wait_for_res, timeout=None):
        if timeout is None:
            timeout = self._comm_timeout

        req_packet = _REQ_PACKET(req, timeout, wait_for_res)
        logging.debug('Process req: {0}'.format(req_packet))

        self._req_q.put(req_packet)
        if wait_for_res:
            try:
                res = self._res_q.get(block=True, timeout=self._process_timeout)
            except (IOError, Empty):
                raise GoFaCommException("Request timed out: {0}".format(req_packet))

            logging.debug('res: {0}'.format(res))

            if res.res_code != GFC.RES_CODES['success']:
                raise GoFaControlException(req_packet, res)

            return res

    @staticmethod
    def _construct_req(code_name, body: str = ""):
        if len(body) < 1:
            _body = []
        else:
            _body = [float(_) for _ in body.split(" ") if _ != ""]
        num_params = len(_body)
        req = struct.pack("HH" + "f" * num_params, num_params, GFC.CMD_CODES[code_name], *_body)
        return req

    @staticmethod
    def _iter_to_str(template, iterable):
        result = ''
        for val in iterable:
            result += template.format(val).rstrip('0').rstrip('.') + ' '
        return result

    @staticmethod
    def _get_pose_body(pose):
        if not isinstance(pose, RigidTransform):
            raise ValueError('Can only parse RigidTransform objects')
        pose = pose.copy()
        pose.position = pose.position * METERS_TO_MM
        body = '{0}{1}'.format(GoFaArm._iter_to_str('{:.1f}', pose.position.tolist()),
                               GoFaArm._iter_to_str('{:.5f}', pose.quaternion.tolist()))
        if pose.configuration is not None:
            body += '{0}'.format(GoFaArm._iter_to_str('{:.5f}', pose.configuration.tolist()))
        return body

    @staticmethod
    def construct_speed_data(tra, rot):
        '''Constructs a speed data tuple that's in the same format as ones used in RAPID.

        Parameters
        ----------
            tra : float
                    translational speed (milimeters per second)
            rot : float
                    rotational speed (degrees per second)

        Returns:
            A tuple of correctly formatted speed data: (tra, rot, tra, rot)
        '''
        return (tra, rot, tra, rot)

    @staticmethod
    def get_v(n):
        '''Gets the corresponding speed data for n as the speed number.

        Parameters
        ----------
            n : int
                    speed number. If n = 100, will return the same speed data as v100 in RAPID

        Returns
        -------
            Corresponding speed data tuple using n as speed number
        '''
        return GoFaArm.construct_speed_data(n, 500)

    @staticmethod
    def from_frame(self):
        return self._from_frame

    @staticmethod
    def to_frame(self):
        return self._to_frame

    def ping(self, wait_for_res=True):
        '''Pings the remote server.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        out : Namedtuple (raw_res, data) from ping command.

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('ping')
        return self._request(req, wait_for_res)

    def get_state(self, raw_res=False):
        '''Get the current state (joint configuration) of this arm.

        Parameters
        ----------
        raw_res : bool, optional
                If True, will return raw_res namedtuple instead of GoFaState
                Defaults to False

        Returns
        -------
        out :
            GoFaState if raw_res is False

            _RES(raw_res, state) namedtuple if raw_res is True

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return GoFaState()

        req = GoFaArm._construct_req('get_joints')
        res = self._request(req, True)

        if res is not None:
            state = message_to_state(res.message)
            if raw_res:
                return _RES(res, state)
            else:
                return state

    def get_torques(self, raw_res=False):
        '''Get the torques (current) of joints.

        Parameters
        ----------
        raw_res : bool, optional
                If True, will return raw_res namedtuple instead of GoFaState
                Defaults to False

        Returns
        -------
        out :
            GoFaState if raw_res is False

            _RES(raw_res, state) namedtuple if raw_res is True

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return GoFaState()

        req = GoFaArm._construct_req('get_torques')
        res = self._request(req, True)
        if res is not None:
            state = message_to_torques(res.message)
            if raw_res:
                return _RES(res, state)
            else:
                return state

    def get_torques_current(self, raw_res=False):
        '''Get the torques (current) of joints.

        Parameters
        ----------
        raw_res : bool, optional
                If True, will return raw_res namedtuple instead of GoFaState
                Defaults to False

        Returns
        -------
        out :
            GoFaState if raw_res is False

            _RES(raw_res, state) namedtuple if raw_res is True

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return GoFaState()

        req = GoFaArm._construct_req('get_torques_current')
        res = self._request(req, True)
        if res is not None:
            state = message_to_torques(res.message)
            if raw_res:
                return _RES(res, state)
            else:
                return state

    def get_pose(self, raw_res=False):
        '''Get the current pose of this arm to base frame of the arm.

        Parameters
        ----------
        raw_res : bool, optional
            If True, will return raw_res namedtuple instead of GoFaState
            Defaults to False

        Returns
        -------
        out :
            RigidTransform if raw_res is False

            _RES(raw_res, pose) namedtuple if raw_res is True

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        # TODO the get value not exactly the same as the value displayed in the robot
        if self._debug:
            return RigidTransform(from_frame=self._from_frame, to_frame=self._to_frame)

        req = GoFaArm._construct_req('get_pose')
        res = self._request(req, True)

        if res is not None:
            pose = message_to_pose(res.message, self._from_frame)
            if raw_res:
                return _RES(res, pose)
            else:
                return pose

    def is_pose_reachable(self, pose):
        '''Check if a given pose is reachable (incurs no kinematic/joint-space limitations and self collisions)

        Parameters
        ----------
        pose : RigidTransform

        Returns
        -------
        bool : True if pose is reachable, False otherwise.

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = GoFaArm._get_pose_body(pose)
        req = GoFaArm._construct_req('is_pose_reachable', body)
        res = self._request(req, True)
        return bool(int(res.message))

    def goto_state(self, state, wait_for_res=True):
        '''Commands the GoFa to goto the given state (joint angles)

        Parameters
        ----------
        state : GoFaState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') if state logging is not enabled and wait_for_res is False

        {
            'time': <flaot>,
            'state': <GoFastate>,
            'res': <namedtuple('_RAW_RES', 'mirror_code res_code message')>
        } otherwise. The time field indicates the duration it took for the arm to complete the motion.

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        body = GoFaArm._iter_to_str('{:.2f}', state.joints)
        req = GoFaArm._construct_req('goto_joints', body)
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)

        if hasattr(self, '_state_logger') and wait_for_res and res is not None:
            if self._debug:
                time = -1.
            else:
                time = float(res.message)
            actual_state = self.get_state()
            self._state_logger.append_time(time)
            self._state_logger.append_expected(state)
            self._state_logger.append_actual(actual_state)
            return {
                'time': time,
                'state': actual_state,
                'res': res
            }

        return res

    def fk(self, state, raw_res=False):
        '''Get the forward _kinematics of this arm to base frame of the arm.

        Parameters
        ----------
        state : GoFaState
        raw_res : bool, optional
            If True, will return raw_res namedtuple instead of GoFaState
            Defaults to False

        Returns
        -------
        out :
            RigidTransform if raw_res is False

            _RES(raw_res, pose) namedtuple if raw_res is True

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return RigidTransform(from_frame=self._from_frame, to_frame=self._to_frame)

        body = GoFaArm._iter_to_str('{:.2f}', state.joints)
        req = GoFaArm._construct_req('fk', body)
        res = self._request(req, True)

        if res is not None:
            pose = message_to_pose(res.message, self._from_frame)
            if raw_res:
                return _RES(res, pose)
            else:
                return pose

    def set_speed_max(self, wait_for_res=True):
        'Set speed of GOFA to its max speed'
        req = GoFaArm._construct_req('set_speed_max')
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)
        return

    def _goto_state_sync(self, state, wait_for_res=True):
        body = GoFaArm._iter_to_str('{:.2f}', state.joints)
        req = GoFaArm._construct_req('goto_joints_sync', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def goto_pose(self, pose, linear=True, relative=False, wait_for_res=True):
        '''Commands the GOFA to goto the given pose

        Parameters
        ----------
        pose : RigidTransform
        linear : bool, optional
            If True, will use MoveL in RAPID to ensure linear path. Otherwise use MoveJ in RAPID, which does not ensure linear path.
            Defaults to True
        relative : bool, optional
            If True, will use goto_pose_relative by computing the delta pose from current pose to target pose.
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') if pose logging is not enabled and wait_for_res is False

        {
            'time': <flaot>,
            'pose': <RigidTransform>,
            'res': <namedtuple('_RAW_RES', 'mirror_code res_code message')>
        } otherwise. The time field indicates the duration it took for the arm to complete the motion.

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if relative:
            cur_pose = self.get_pose()
            delta_pose = cur_pose.inverse() * pose
            tra = delta_pose.translation
            rot = np.rad2deg(delta_pose.euler_angles)
            res = self.goto_pose_delta(tra, rot, wait_for_res=wait_for_res)
        else:
            body = GoFaArm._get_pose_body(pose)
            if linear:
                cmd = 'goto_pose_linear'
            else:
                cmd = 'goto_pose'
            req = GoFaArm._construct_req(cmd, body)
            res = self._request(req, wait_for_res, timeout=self._motion_timeout)

        if hasattr(self, '_pose_logger') and wait_for_res and res is not None:
            if self._debug:
                time = -1.
            else:
                time = float(res.message)
            actual_pose = self.get_pose()
            self._pose_logger.append_time(time)
            self._pose_logger.append_expected(pose)
            self._pose_logger.append_actual(actual_pose)
            return {
                'time': time,
                'pose': actual_pose,
                'res': res
            }

        return res

    def _goto_pose_sync(self, pose, wait_for_res=True):
        body = GoFaArm._get_pose_body(pose)
        req = GoFaArm._construct_req('goto_pose_sync', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def goto_pose_delta(self, translation, rotation=None, wait_for_res=True):
        '''Goto a target pose by transforming the current pose using the given translation and rotation

        Parameters
        ----------
        translation : list-like with axis_length 3
            The translation vector (x, y, z) in meters.
        rotation : list-like with axis_length 3, optional
            The euler angles of given rotation in degrees.
            Defaults to 0 degrees - no rotation.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        translation = [val * METERS_TO_MM for val in translation]
        translation_str = GoFaArm._iter_to_str('{:.1f}', translation)
        rotation_str = ''
        if rotation is not None:
            rotation_str = GoFaArm._iter_to_str('{:.5f}', rotation)

        body = translation_str + rotation_str
        req = GoFaArm._construct_req('goto_pose_delta', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def set_tool(self, pose, wait_for_res=True):
        '''Sets the Tool Center Point (TCP) of the arm using the given pose.

        Parameters
        ----------
        pose : RigidTransform
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = GoFaArm._get_pose_body(pose)
        req = GoFaArm._construct_req('set_tool', body)
        self._last_sets['tool'] = pose
        return self._request(req, wait_for_res)

    def set_speed(self, speed_data, wait_for_res=True):
        '''Sets the target speed of the arm's movements.

        Parameters
        ----------
        speed_data : list-like with axis_length 4
            Specifies the speed data that will be used by RAPID when executing motions.
            Should be generated using GoFaRobot.get_v
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = GoFaArm._iter_to_str('{:.2f}', speed_data)
        req = GoFaArm._construct_req('set_speed', body)
        self._last_sets['speed'] = speed_data
        return self._request(req, wait_for_res)

    def set_zone(self, zone_data, wait_for_res=True):
        '''Goto a target pose by transforming the current pose using the given translation and rotation

        Parameters
        ----------
        speed_data : list-like with axis_length 4
            Specifies the speed data that will be used by RAPID when executing motions.
            Should be generated using GoFaRobot.get_v
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        pm = zone_data['point_motion']
        data = (pm,) + zone_data['values']
        body = GoFaArm._iter_to_str('{:2f}', data)
        req = GoFaArm._construct_req('set_zone', body)
        self._last_sets['zone'] = zone_data
        return self._request(req, wait_for_res)

    def move_circular(self, center_pose, target_pose, wait_for_res=True):
        '''Goto a target pose by following a circular path around the center_pose

        Parameters
        ----------
        center_pose : RigidTransform
            Pose for the center of the circle for circula movement.
        target_pose : RigidTransform
            Target pose
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        body_set_circ_point = GoFaArm._get_pose_body(center_pose)
        body_move_by_circ_point = GoFaArm._get_pose_body(target_pose)

        req_set_circ_point = GoFaArm._construct_req('set_circ_point', body_set_circ_point)
        req_move_by_circ_point = GoFaArm._construct_req('move_by_circ_point', body_move_by_circ_point)

        res_set_circ_point = self._request(req_set_circ_point, True)
        if res_set_circ_point is None:
            logging.error("Set circular point failed. Skipping move circular!")
            return None
        else:
            return self._request(req_move_by_circ_point, wait_for_res, timeout=self._motion_timeout)

    def movetstate_cont(self, statelist, is_add_all=True, wait_for_res=True) -> bool:
        """
        add states to buffer, execute, and clear

        :param statelist:
        :param wait_for_res:
        :return:

        author: weiwei
        date: 20191024
        Revised by hao chen 20230831
        """
        self.buffer_j_clear(wait_for_res)
        if is_add_all:
            self.buffer_j_add_all2(statelist, wait_for_res=wait_for_res)
        else:
            self.buffer_j_add_all(statelist, wait_for_res=wait_for_res)
        exec_result = self.buffer_j_move(wait_for_res)
        self.buffer_j_clear(wait_for_res)
        return exec_result

    def movetstate_sgl(self, state, wait_for_res=True):
        """
        add states to buffer, execute, and clear

        :param state:
        :param wait_for_res:
        :return:

        author: weiwei
        date: 20191024
        """

        self.goto_state(state, wait_for_res)

    def buffer_c_add_single(self, pose, wait_for_res=True):
        '''Add single pose to the linear movement buffer in RAPID

        Parameters
        ----------
        pose : RigidTransform
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = GoFaArm._get_pose_body(pose)
        req = GoFaArm._construct_req('buffer_c_add', body)
        return self._request(req, wait_for_res)

    def buffer_c_add_all(self, pose_list, wait_for_res=True):
        '''Add a list of poses to the linear movement buffer in RAPID

        Parameters
        ----------
        pose_list : list of RigidTransforms
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        ress = [self.buffer_c_add_single(pose, wait_for_res) for pose in pose_list]
        return ress

    def buffer_c_clear(self, wait_for_res=True):
        '''Clears the linear movement buffer in RAPID

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_c_clear')
        return self._request(req, wait_for_res)

    def buffer_c_size(self, raw_res=False):
        '''Gets the current linear movement buffer size.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_c_size')
        res = self._request(req, True)

        if res is not None:
            try:
                size = int(res.message)
                if raw_res:
                    return _RES(res, size)
                else:
                    return size
            except Exception as e:
                logging.error(e)

    def buffer_c_move(self, wait_for_res=True):
        '''Executes the linear movement buffer

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_c_move')
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def buffer_j_add_single(self, state, wait_for_res=True):
        '''Add single state to the joint movement buffer in RAPID

        Parameters
        ----------
        state : GoFaState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = GoFaArm._iter_to_str('{:.2f}', state.joints)
        req = GoFaArm._construct_req('buffer_j_add', body)
        return self._request(req, wait_for_res)

    def buffer_j_add_all(self, state_list, wait_for_res=True):
        '''Add a list of states to the joint movement buffer in RAPID

        Parameters
        ----------
        state_list : list of GoFaState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        ress = [self.buffer_j_add_single(state, wait_for_res) for state in state_list]
        return ress

    def buffer_j_add_single2(self, state_list, wait_for_res):
        body = GoFaArm._iter_to_str('{:.2f}', [v for state in state_list for v in state.joints])
        req = GoFaArm._construct_req('buffer_j_add_all', f"{len(state_list)} " + body)
        return self._request(req, wait_for_res)

    def buffer_j_add_all2(self, state_list, msg_len=35, wait_for_res=False):
        ress = [self.buffer_j_add_single2(_, wait_for_res) for _ in
                [state_list[x:x + msg_len] for x in range(0, len(state_list), msg_len)]]
        return ress

    def buffer_j_clear(self, wait_for_res=True):
        '''Clears the joint movement buffer in RAPID

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_j_clear')
        return self._request(req, wait_for_res)

    def buffer_j_size(self, raw_res=False):
        '''Gets the current joint movement buffer size.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_j_size')
        res = self._request(req, True)

        if res is not None:
            try:
                size = int(res.message)
                if raw_res:
                    return _RES(res, size)
                else:
                    return size
            except Exception as e:
                logging.error(e)

    def buffer_j_move(self, wait_for_res=True):
        '''Executes the joint movement buffer

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        req = GoFaArm._construct_req('buffer_j_move')
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)
        if res is not None:
            return bool(int(res.message))
        else:
            return None

    def write_handcamimg_ftp(self):
        """
        triger a camera image and send it to the ftp server run on this machine

        :return:

        author: weiwei
        date: 20200519
        """

        req = GoFaArm._construct_req('write_handcamimg_ftp')
        return self._request(req, wait_for_res=True)

    def toggle_vacuum(self, toggletag=True):
        """
        toggle vacuum on and off, toggletag is True or False

        :return:

        author: weiwei
        date: 20200519
        """

        if toggletag:
            req = GoFaArm._construct_req('set_vacuum_on')
            return self._request(req, wait_for_res=True)
        else:
            req = GoFaArm._construct_req('set_vacuum_off')
            return self._request(req, wait_for_res=True)

    def get_pressure(self):
        """
        get the pressure of the vacuum cup

        :return:

        author: weiwei
        date: 20200519
        """

        req = GoFaArm._construct_req('get_pressure')
        res = self._request(req, wait_for_res=True)
        pressure = float(res.message)  # kpa

        return pressure

    def open_gripper(self, no_wait=False, wait_for_res=True):
        '''Opens the gripper to the target_width

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = GoFaArm._construct_req('open_gripper', '')
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def close_gripper(self, force=GFC.MAX_GRIPPER_FORCE, width=0., no_wait=False,
                      wait_for_res=True):
        '''Closes the gripper as close to 0 as possible with maximum force.

        Parameters
        ----------
        force : float, optional
            Sets the corresponding gripping force in Newtons.
            Defaults to 20, which is the maximum grip force.
        width : float, optional
            Sets the target width of gripper close motion in m. Cannot be greater than max gripper width.
            Defaults to 0.
        no_wait : bool, optional
            If True, the RAPID server will continue without waiting for the gripper to reach its target width
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if force < 0 or force > GFC.MAX_GRIPPER_FORCE:
            raise ValueError(
                "Gripper force can only be between {} and {}. Got {}.".format(0, GFC.MAX_GRIPPER_FORCE, force))
        if width < 0 or width > GFC.MAX_GRIPPER_WIDTH:
            raise ValueError(
                "Gripper width can only be between {} and {}. Got {}.".format(0, GFC.MAX_GRIPPER_WIDTH, width))

        width = METERS_TO_MM * width
        body = GoFaArm._iter_to_str('{0:.1f}', [force, width] + ([0] if no_wait else []))
        req = GoFaArm._construct_req('close_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def move_gripper(self, width, no_wait=False, wait_for_res=True):
        '''Moves the gripper to the given width in meters.

        Parameters
        ----------
        width : float
            Target width in meters
        no_wait : bool, optional
            If True, the RAPID server will continue without waiting for the gripper to reach its target width
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        lst = [width * METERS_TO_MM]
        if no_wait:
            lst.append(0)
        body = GoFaArm._iter_to_str('{0:.1f}', lst)
        req = GoFaArm._construct_req('move_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def calibrate_gripper(self, max_speed=None, hold_force=None, phys_limit=None, wait_for_res=True):
        '''Calibrates the gripper.

        Parameters
        ----------
        max_speed : float, optional
            Max speed of the gripper in mm/s.
            Defaults to None. If None, will use maximum speed in RAPID.
        hold_force : float, optional
            Hold force used by the gripper in N.
            Defaults to None. If None, will use maximum force the gripper can provide (20N).
        phys_limit : float, optional
            The maximum opening of the gripper.
            Defaults to None. If None, will use maximum opening the gripper can provide (25mm).
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.

        Notes
        -----
        All 3 values must be provided, or they'll all default to None.
        '''
        if None in (max_speed, hold_force, phys_limit):
            body = ''
        else:
            body = self._iter_to_str('{:.1f}', [data['max_speed'], data['hold_force'], data['phys_limit']])
        req = GoFaArm._construct_req('calibrate_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def set_gripper_force(self, force, wait_for_res=True):
        '''Sets the gripper hold force

        Parameters
        ----------
        force : float
            Hold force by the gripper in N.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = self._iter_to_str('{:.1f}', [force])
        req = GoFaArm._construct_req('set_gripper_force', body)
        self._last_sets['gripper_force'] = force
        return self._request(req, wait_for_res)

    def set_gripper_max_speed(self, max_speed, wait_for_res=True):
        '''Sets the gripper max speed

        Parameters
        ----------
        max_speed : float
            In mm/s.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        '''
        body = self._iter_to_str('{:1f}', [max_speed])
        req = GoFaArm._construct_req('set_gripper_max_speed', body)
        self._last_sets['gripper_max_speed'] = max_speed
        return self._request(req, wait_for_res)

    def get_gripper_width(self, raw_res=False):
        '''Get width of current gripper in meters.

        Parameters
        ----------
        raw_res : bool, optional

        Returns
        -------
        Width in meters if raw_res is False
        namedtuple('_RES', 'res width') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = GoFaArm._construct_req('get_gripper_width')
        res = self._request(req, wait_for_res=True)

        if self._debug:
            return -1.

        width = float(res.message) * MM_TO_METERS
        if raw_res:
            return _RES(res, width)
        else:
            return width

    def reset_home(self, wait_for_res=True):
        '''Resets the arm to home using joints

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        GoFaCommException
            If communication times out or socket error.
        GoFaCommException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = GoFaArm._construct_req('reset_home')
        return self._request(req, wait_for_res)


if __name__ == '__main__':
    logging.getLogger().setLevel(GFC.LOGGING_LEVEL)