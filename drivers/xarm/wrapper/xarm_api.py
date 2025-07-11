#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2018, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

from ..x3 import XArm


class XArmAPI(object):
    def __init__(self, port=None, is_radian=False, do_not_open=False, **kwargs):
        """
        The API wrapper of xArm
        Note: Orientation of attitude angle
            roll: rotate around the X axis
            pitch: rotate around the Y axis
            yaw: rotate around the Z axis

        :param port: ip-address(such as '192.168.1.185')
            Note: this parameter is required if parameter do_not_open is False
        :param is_radian: set the default unit is radians or not, default is False
            Note: (aim of design)
                1. Default value for unified interface parameters
                2: Unification of the external unit system
                3. For compatibility with previous interfaces
            Note: the conversion of degree (°) and radians (rad)
                * 1 rad == 57.29577951308232 °
                * 1 ° == 0.017453292519943295 rad
                * 1 rad/s == 57.29577951308232 °/s
                * 1 °/s == 0.017453292519943295 rad/s
                * 1 rad/s^2 == 57.29577951308232 °/s^2
                * 1 °/s^2 == 0.017453292519943295 rad/s^2
                * 1 rad/s^3 == 57.29577951308232 °/s^3
                * 1 °/s^3 == 0.017453292519943295 rad/s^3
            Note: This parameter determines the value of the property self.default_is_radian
            Note: This parameter determines the default value of the interface with the is_radian/input_is_radian/return_is_radian parameter
               The list of affected interfaces is as follows:
                    1. method: get_position
                    2. method: set_position
                    3. method: get_servo_angle
                    4. method: set_servo_angle
                    5. method: set_servo_angle_j
                    6. method: move_gohome
                    7. method: reset
                    8. method: set_tcp_offset
                    9. method: set_joint_jerk
                    10. method: set_joint_maxacc
                    11. method: get_inverse_kinematics
                    12. method: get_forward_kinematics
                    13. method: is_tcp_limit
                    14. method: is_joint_limit
                    15. method: get_params
                    16: method: move_arc_lines
                    17: method: move_circle
                    18: method: set_servo_cartesian
            Note: This parameter determines the default return type for some interfaces (such as the position, velocity, and acceleration associated with the return angle arc).
                The affected attributes are as follows:
                    1. property: position
                    2. property: last_used_position
                    3. property: angles
                    4. property: last_used_angles
                    5. property: last_used_joint_speed
                    6. property: last_used_joint_acc
                    7. property: tcp_offset
        :param do_not_open: do not open, default is False, if true, you need to manually call the connect interface.
        :param kwargs: keyword parameters, generally do not need to set
            axis: number of axes, required only when using a serial port connection, default is 7
            baudrate: serial baudrate, invalid, reserved.
            timeout: serial timeout, invalid, reserved.
            filters: serial port filters, invalid, reserved.
            check_tcp_limit: check the tcp param value out of limit or not, default is False
                Note: only check the param roll/pitch/yaw of the interface `set_position`/`move_arc_lines`
            check_joint_limit: check the joint param value out of limit or not, default is True
                Note: only check the param angle of the interface `set_servo_angle` and the param angles of the interface `set_servo_angle_j`
            check_cmdnum_limit: check the cmdnum out of limit or not, default is True
            max_cmdnum: max cmdnum, default is 256
                Note: only available in the param `check_cmdnum_limit` is True
            check_is_ready: check if the arm is in motion, default is True
        """
        self._arm = XArm(port=port,
                         is_radian=is_radian,
                         do_not_open=do_not_open,
                         instance=self,
                         **kwargs)
        self.__attr_alias_map = {
            'get_ik': self.get_inverse_kinematics,
            'get_fk': self.get_forward_kinematics,
            'set_sleep_time': self.set_pause_time,
            'register_maable_mtbrake_changed_callback': self.register_mtable_mtbrake_changed_callback,
            'release_maable_mtbrake_changed_callback': self.release_mtable_mtbrake_changed_callback,
            'position_offset': self.tcp_offset,
            'get_gpio_digital': self.get_tgpio_digital,
            'set_gpio_digital': self.set_tgpio_digital,
            'get_gpio_analog': self.get_tgpio_analog,
            'set_fense_mode': self.set_fence_mode,
            'get_suction_cup': self.get_vacuum_gripper,
            'set_suction_cup': self.set_vacuum_gripper,
        }

    def __getattr__(self, item):
        if item in self.__attr_alias_map.keys():
            return self.__attr_alias_map[item]
        raise AttributeError('\'{}\' has not attribute \'{}\''.format(self.__class__.__name__, item))

    @property
    def arm(self):
        return self._arm

    @property
    def core(self):
        """
        Core layer API, set only for advanced developers, please do not use
        Ex:
            self.core.move_line(...)
            self.core.move_lineb(...)
            self.core.move_joint(...)
            ...
        """
        return self._arm.arm_cmd

    @property
    def count(self):
        """
        Counter val
        """
        return self._arm.count

    @property
    def realtime_tcp_speed(self):
        """
        The real time speed of tcp motion, only available if version > 1.2.11

        :return: real time speed (mm/s)
        """
        return self._arm.realtime_tcp_speed

    @property
    def realtime_joint_speeds(self):
        """
        The real time speed of joint motion, only available if version > 1.2.11

        :return: [joint-1-speed(°/s or rad/s), ...., joint-7-speed(°/s or rad/s)]
        """
        return self._arm.realtime_joint_speeds

    @property
    def gpio_reset_config(self):
        """
        The gpio reset enable config
        :return: [cgpio_reset_enable, tgpio_reset_enable] 
        """
        return self._arm.gpio_reset_config

    @property
    def version_number(self):
        """
        Frimware version number
        
        :return: (major_version_number, minor_version_number, revision_version_number)
        """
        return self._arm.version_number

    @property
    def connected(self):
        """
        Connection status
        """
        return self._arm.connected

    @property
    def default_is_radian(self):
        """
        The default unit is radians or not
        """
        return self._arm.default_is_radian

    @property
    def version(self):
        """
        xArm version
        """
        return self._arm.version

    @property
    def sn(self):
        """
        xArm sn
        """
        return self._arm.sn
    
    @property
    def control_box_sn(self):
        """
        Control box sn
        """
        return self._arm.control_box_sn

    @property
    def position(self):
        """
        Cartesion position
        Note:
            1. If self.default_is_radian is True, the returned value (only roll/pitch/yaw) is in radians

        return: [x(mm), y(mm), z(mm), roll(° or rad), pitch(° or rad), yaw(° or rad)]
        """
        return self._arm.position

    @property
    def last_used_position(self):
        """
        The last used cartesion position, default value of parameter x/y/z/roll/pitch/yaw of interface set_position
        Note:
            1. If self.default_is_radian is True, the returned value (only roll/pitch/yaw) is in radians
            2. self.set_position(x=300) <==> self.set_position(x=300, *last_used_position[1:])
            2. self.set_position(roll=-180) <==> self.set_position(x=self.last_used_position[:3], roll=-180, *self.last_used_position[4:])

        :return: [x(mm), y(mm), z(mm), roll(° or rad), pitch(° or rad), yaw(° or rad)]
        """
        return self._arm.last_used_position

    @property
    def tcp_jerk(self):
        """
        Tcp jerk

        :return: jerk (mm/s^3)
        """
        return self._arm.tcp_jerk

    @property
    def tcp_speed_limit(self):
        """
        Tcp speed limit, only available in socket way and enable_report is True and report_type is 'rich'

        :return: [min_tcp_speed(mm/s), max_tcp_speed(mm/s)]
        """
        return self._arm.tcp_speed_limit

    @property
    def tcp_acc_limit(self):
        """
        Tcp acceleration limit, only available in socket way and enable_report is True and report_type is 'rich' 
        
        :return: [min_tcp_acc(mm/s^2), max_tcp_acc(mm/s^2)]
        """
        return self._arm.tcp_acc_limit

    @property
    def last_used_tcp_speed(self):
        """
        The last used cartesion speed, default value of parameter speed of interface set_position/move_circle

        :return: speed (mm/s)
        """
        return self._arm.last_used_tcp_speed

    @property
    def last_used_tcp_acc(self):
        """
        The last used cartesion acceleration, default value of parameter mvacc of interface set_position/move_circle

        :return: acceleration (mm/s^2)
        """
        return self._arm.last_used_tcp_acc

    @property
    def angles(self):
        """
        Servo angles
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: [angle1(° or rad), angle2(° or rad), ..., anglen7(° or rad)]
        """
        return self._arm.angles

    @property
    def joint_jerk(self):
        """
        Joint jerk
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: jerk (°/s^3 or rad/s^3)
        """
        return self._arm.joint_jerk

    @property
    def joint_speed_limit(self):
        """
        Joint speed limit,  only available in socket way and enable_report is True and report_type is 'rich'
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: [min_joint_speed(°/s or rad/s), max_joint_speed(°/s or rad/s)]
        """
        return self._arm.joint_speed_limit

    @property
    def joint_acc_limit(self):
        """
        Joint acceleration limit, only available in socket way and enable_report is True and report_type is 'rich'
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: [min_joint_acc(°/s^2 or rad/s^2), max_joint_acc(°/s^2 or rad/s^2)]
        """
        return self._arm.joint_acc_limit

    @property
    def last_used_angles(self):
        """
        The last used servo angles, default value of parameter angle of interface set_servo_angle
        Note:
            1. If self.default_is_radian is True, the returned value is in radians
            2. self.set_servo_angle(servo_id=1, angle=75) <==> self.set_servo_angle(angle=[75] + self.last_used_angles[1:])
            3. self.set_servo_angle(servo_id=5, angle=30) <==> self.set_servo_angle(angle=self.last_used_angles[:4] + [30] + self.last_used_angles[5:])

        :return: [angle1(° or rad), angle2(° or rad), ..., angle7(° or rad)]
        """
        return self._arm.last_used_angles

    @property
    def last_used_joint_speed(self):
        """
        The last used joint speed, default value of parameter speed of interface set_servo_angle
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: speed (°/s or rad/s)
        """
        return self._arm.last_used_joint_speed

    @property
    def last_used_joint_acc(self):
        """
        The last used joint acceleration, default value of parameter mvacc of interface set_servo_angle
        Note:
            1. If self.default_is_radian is True, the returned value is in radians

        :return: acceleration (°/s^2 or rad/s^2)
        """
        return self._arm.last_used_joint_acc

    @property
    def tcp_offset(self):
        """
        Cartesion position offset, only available in socket way and enable_report is True
        Note:
            1. If self.default_is_radian is True, the returned value(roll_offset/pitch_offset/yaw_offset) is in radians

        :return: [x_offset(mm), y_offset(mm), z_offset(mm), roll_offset(° or rad), pitch_offset(° or rad), yaw_offset(° or rad)]
        """
        return self._arm.position_offset

    @property
    def world_offset(self):
        """
        Base coordinate offset, only available if version > 1.2.11

        Note:
            1. If self.default_is_radian is True, the returned value(roll_offset/pitch_offset/yaw_offset) is in radians

        :return: [x_offset(mm), y_offset(mm), z_offset(mm), roll_offset(° or rad), pitch_offset(° or rad), yaw_offset(° or rad)]
        """
        return self._arm.world_offset

    @property
    def state(self):
        """
        xArm state

        :return:
            1: in motion
            2: sleeping
            3: suspended
            4: stopping
        """
        return self._arm.state

    @property
    def mode(self):
        """
        xArm mode,only available in socket way and  enable_report is True

        :return:
            0: position control mode
            1: servo motion mode
            2: joint teaching mode
            3: cartesian teaching mode (invalid)
        """
        return self._arm.mode

    @property
    def is_simulation_robot(self):
        """
        Is simulation rbt_s not not
        """
        return self._arm.is_simulation_robot

    @property
    def joints_torque(self):
        """
        Joints torque, only available in socket way and  enable_report is True and report_type is 'rich'

        :return: [joint-1, ....]
        """
        return self._arm.joints_torque

    @property
    def tcp_load(self):
        """
        xArm tcp load, only available in socket way and  enable_report is True and report_type is 'rich'

        :return: [weight, center of gravity]
            such as: [weight(kg), [x(mm), y(mm), z(mm)]]
        """
        return self._arm.tcp_load

    @property
    def collision_sensitivity(self):
        """
        The sensitivity value of collision, only available in socket way and  enable_report is True and report_type is 'rich'

        :return: 0~5
        """
        return self._arm.collision_sensitivity

    @property
    def teach_sensitivity(self):
        """
        The sensitivity value of drag and teach, only available in socket way and  enable_report is True and report_type is 'rich'

        :return: 1~5
        """
        return self._arm.teach_sensitivity

    @property
    def motor_brake_states(self):
        """
        Motor brake state list, only available in socket way and  enable_report is True and report_type is 'rich'
        Note:
            For a rbt_s with a number of axes n, only the first n states are valid, and the latter are reserved.

        :return: [motor-1-brake-state, ..., motor-7-brake-state, reserved]
            motor-{i}-brake-state:
                0: enable
                1: disable
        """
        return self._arm.motor_brake_states

    @property
    def motor_enable_states(self):
        """
        Motor enable state list, only available in socket way and  enable_report is True and report_type is 'rich'
        Note:
            For a rbt_s with a number of axes n, only the first n states are valid, and the latter are reserved.

        :return: [motor-1-enable-state, ..., motor-7-enable-state, reserved]
            motor-{i}-enable-state:
                0: disable
                1: enable
        """
        return self._arm.motor_enable_states

    @property
    def temperatures(self):
        """
        Motor temperature, only available if version > 1.2.11

        :return: [motor-1-temperature, ..., motor-7-temperature]
        """
        return self._arm.temperatures

    @property
    def has_err_warn(self):
        """
        Contorller have an error or warning or not

        :return: True/False
        """
        return self._arm.has_err_warn

    @property
    def has_error(self):
        """
        Controller have an error or not
        """
        return self._arm.has_error

    @property
    def has_warn(self):
        """
        Controller have an warnning or not
        """
        return self._arm.has_warn

    @property
    def error_code(self):
        """
        Controller error code. See Chapter 7 of the xArm User Manual for details.
        """
        return self._arm.error_code

    @property
    def warn_code(self):
        """
        Controller warn code. See Chapter 7 of the xArm User Manual for details.
        """
        return self._arm.warn_code

    @property
    def cmd_num(self):
        """
        Number of command caches in the controller
        """
        return self._arm.cmd_num

    @property
    def device_type(self):
        """
        Device type, only available in socket way and  enable_report is True and report_type is 'rich'
        """
        return self._arm.device_type

    @property
    def axis(self):
        """
        Axis number, only available in socket way and enable_report is True and report_type is 'rich'
        """
        return self._arm.axis

    @property
    def master_id(self):
        """
        Master id, only available in socket way and enable_report is True and report_type is 'rich'
        """
        return self._arm.master_id

    @property
    def slave_id(self):
        """
        Slave id, only available in socket way and enable_report is True and report_type is 'rich'
        """
        return self._arm.slave_id

    @property
    def gravity_direction(self):
        """
        gravity direction, only available in socket way and enable_report is True and report_type is 'rich'
        :return:
        """
        return self._arm.gravity_direction

    @property
    def servo_codes(self):
        """
        Servos status and error_code
        :return: [
            [servo-1-status, servo-1-code],
            ...,
            [servo-7-status, servo-7-code], 
            [tool-gpio-status, tool-gpio-code]
        ]
        """
        return self._arm.servo_codes

    @property
    def voltages(self):
        """
        Servos voltage
        
        :return: [servo-1-voltage, ..., servo-7-voltage]
        """
        return self._arm.voltages

    @property
    def currents(self):
        """
        Servos electric current
        
        :return: [servo-1-current, ..., servo-7-current]
        """
        return self._arm.currents

    @property
    def cgpio_states(self):
        """
        Controller gpio state
        
        :return: states
            states[0]: contorller gpio module state
                states[0] == 0: normal
                states[0] == 1: wrong
                states[0] == 6: communication failure
            states[1]: controller gpio module error code
                states[1] == 0: normal
                states[1] != 0: error code
            states[2]: digital input functional gpio state
                Note: digital-i-input functional gpio state = states[2] >> i & 0x01
            states[3]: digital input configuring gpio state
                Note: digital-i-input configuring gpio state = states[3] >> i & 0x01
            states[4]: digital output functional gpio state
                Note: digital-i-output functional gpio state = states[4] >> i & 0x01
            states[5]: digital output configuring gpio state
                Note: digital-i-output configuring gpio state = states[5] >> i & 0x01
            states[6]: analog-0 input value
            states[7]: analog-1 input value
            states[8]: analog-0 output value
            states[9]: analog-1 output value
            states[10]: digital input functional info, [digital-0-input-functional-mode, ... digital-7-input-functional-mode]
            states[11]: digital output functional info, [digital-0-output-functional-mode, ... digital-7-output-functional-mode]
        """
        return self._arm.cgpio_states

    @property
    def self_collision_params(self):
        """
        Self collision params
        
        :return: params
            params[0]: self collision detection or not
            params[1]: self collision tool type
            params[2]: self collision model params
        """
        return self._arm.self_collision_params

    def connect(self, port=None, baudrate=None, timeout=None, axis=None, **kwargs):
        """
        Connect to xArm

        :param port: port name or the ip address, default is the value when initializing an instance
        :param baudrate: baudrate, only available in serial way, default is the value when initializing an instance
        :param timeout: timeout, only available in serial way, default is the value when initializing an instance
        :param axis: number of axes, required only when using a serial port connection, default is 7
        """
        self._arm.connect(port=port, baudrate=baudrate, timeout=timeout, axis=axis, **kwargs)

    def disconnect(self):
        """
        Disconnect
        """
        self._arm.disconnect()

    def send_cmd_sync(self, command=None):
        """
        Send cmd and wait (only waiting the cmd response, not waiting for the movement)
        Note:
            1. Some command depends on self.default_is_radian

        :param command:
            'G1': 'set_position(MoveLine): G1 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw} F{speed} Q{acc} T{mvtime}'
            'G2': 'move_circle: G2 X{x1} Y{y1} Z{z1} A{roll1} B{pitch1} C{yaw1} I{x2} J{y2} K{z2} L{roll2} M{pitch2} N{yaw2} F{speed} Q{acc} T{mvtime}'
            'G4': 'set_pause_time: G4 T{second}'
            'G7': 'set_servo_angle: G7 I{servo_1} J{servo_2} K{servo_3} L{servo_4} M{servo_5} N{servo_6} O{servo_7} F{speed} Q{acc} T{mvtime}'
            'G8': 'move_gohome: G8 F{speed} Q{acc} T{mvtime}'
            'G9': 'set_position(MoveArcLine): G9 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw} R{radius} F{speed} Q{acc} T{mvtime}'
            'G11': 'set_servo_angle_j: G11 I{servo_1} J{servo_2} K{servo_3} L{servo_4} M{servo_5} N{servo_6} O{servo_7} F{speed} Q{acc} T{mvtime}'
            'H1': 'get_version: H1'
            'H11': 'motion_enable: H11 I{servo_id} V{enable}'
            'H12': 'set_state: H12 V{state}'
            'H13': 'get_state: H13'
            'H14': 'get_cmdnum: H14'
            'H15': 'get_err_warn_code: H15'
            'H16': 'clean_error: H16'
            'H17': 'clean_warn: H17'
            'H18': 'set_servo_attach/set_servo_detach: H18 I{servo_id} V{1: enable(detach), 0: disable(attach)}'
            'H19': 'set_mode: H19 V{mode}'
            'H31': 'set_tcp_jerk: H31 V{jerk(mm/s^3)}'
            'H32': 'set_tcp_maxacc: H32 V{maxacc(mm/s^2)}'
            'H33': 'set_joint_jerk: H33 V{jerk(°/s^3 or rad/s^3)}'
            'H34': 'set_joint_maxacc: H34 {maxacc(°/s^2 or rad/s^2)}'
            'H35': 'set_tcp_offset: H35 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw}'
            'H36': 'set_tcp_load: H36 I{weight} J{center_x} K{center_y} L{center_z}'
            'H37': 'set_collision_sensitivity: H37 V{sensitivity}'
            'H38': 'set_teach_sensitivity: H38 V{sensitivity}'
            'H39': 'clean_conf: H39'
            'H40': 'save_conf: H40'
            'H41': 'get_position: H41'
            'H42': 'get_servo_angle: H42'
            'H43': 'get_inverse_kinematics: H43 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw}'
            'H44': 'get_forward_kinematics: H44 I{servo_1} J{servo_2} K{servo_3} L{servo_4} M{servo_5} N{servo_6} O{servo_7}'
            'H45': 'is_joint_limit: H45 I{servo_1} J{servo_2} K{servo_3} L{servo_4} M{servo_5} N{servo_6} O{servo_7}'
            'H46': 'is_tcp_limit: H46 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw}'
            'H51': 'set_gravity_direction: H51 X{x} Y{y} Z{z}'
            'H106': 'get_servo_debug_msg: H106'
            'M116': 'set_gripper_enable: M116 V{enable}'
            'M117': 'set_gripper_mode: M117 V{mode}'
            'M119': 'get_gripper_position: M119'
            'M120': 'set_gripper_position: M120 V{pos}'
            'M121': 'set_gripper_speed: M116 V{speed}'
            'M125': 'get_gripper_err_code: M125'
            'M126': 'clean_gripper_error: M126'
            'M131': 'get_tgpio_digital: M131'
            'M132': 'set_tgpio_digital: M132 I{ionum} V{value}'
            'M133': 'get_tgpio_analog, default ionum=0: M133 I{ionum=0}'
            'M134': 'get_tgpio_analog, default ionum=1: M134 I{ionum=1}'
            'C131': 'get_cgpio_digital: C131'
            'C132': 'get_cgpio_analog, default ionum=0: C132 I{ionum=0}'
            'C133': 'get_cgpio_analog, default ionum=1: C133 I{ionum=1}'
            'C134': 'set_cgpio_digital: C134 I{ionum} V{value}'
            'C135': 'set_cgpio_analog, default ionum=0: C135 I{ionum=0} V{value}'
            'C136': 'set_cgpio_analog, default ionum=1: C136 I{ionum=1} V{value}'
            'C137': 'set_cgpio_digital_input_function: C137 I{ionum} V{fun}'
            'C138': 'set_cgpio_digital_output_function: C138 I{ionum} V{fun}'
            'C139': 'get_cgpio_state: C139'
        :return: code or tuple((code, ...))
            code: See the API code documentation for details.
        """
        return self._arm.send_cmd_sync(command=command)

    def get_position(self, is_radian=None):
        """
        Get the cartesian position
        Note:
            1. If the value(roll/pitch/yaw) you want to return is an radian unit, please set the parameter is_radian to True
                ex: code, pos = arm.get_position(is_radian=True)

        :param is_radian: the returned value (only roll/pitch/yaw) is in radians or not, default is self.default_is_radian
        :return: tuple((code, [x, y, z, roll, pitch, yaw])), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_position(is_radian=is_radian)

    def set_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, radius=None,
                     speed=None, mvacc=None, mvtime=None, relative=False, is_radian=None,
                     wait=False, timeout=None, **kwargs):
        """
        Set the cartesian position, the API will modify self.last_used_position value
        Note:
            1. If it is xArm5, ensure that the current robotic arm has a roll value of 180° or π rad and has a roll value of 0 before calling this interface.
            2. If it is xArm5, roll must be set to 180° or π rad, pitch must be set to 0
            3. If the parameter(roll/pitch/yaw) you are passing is an radian unit, be sure to set the parameter is_radian to True.
                ex: code = arm.set_position(x=300, y=0, z=200, roll=-3.14, pitch=0, yaw=0, is_radian=True)
            4. If you want to wait for the rbt_s to complete this action and then return, please set the parameter wait to True.
                ex: code = arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, is_radian=False, wait=True)
            5. This interface is only used in the base coordinate system.

        :param x: cartesian position x, (unit: mm), default is self.last_used_position[0]
        :param y: cartesian position y, (unit: mm), default is self.last_used_position[1]
        :param z: cartesian position z, (unit: mm), default is self.last_used_position[2]
        :param roll: rotate around the X axis, (unit: rad if is_radian is True else °), default is self.last_used_position[3]
        :param pitch: rotate around the Y axis, (unit: rad if is_radian is True else °), default is self.last_used_position[4]
        :param yaw: rotate around the Z axis, (unit: rad if is_radian is True else °), default is self.last_used_position[5]
        :param radius: move radius, if radius is None or radius less than 0, will MoveLine, else MoveArcLine
            MoveLine: Linear motion
                ex: code = arm.set_position(..., radius=None)
            MoveArcLine: Linear arc motion with interpolation
                ex: code = arm.set_position(..., radius=0)
                Note: Need to set radius>=0
        :param speed: move speed (mm/s, rad/s), default is self.last_used_tcp_speed
        :param mvacc: move acceleration (mm/s^2, rad/s^2), default is self.last_used_tcp_acc
        :param mvtime: 0, reserved
        :param relative: relative move or not
        :param is_radian: the roll/pitch/yaw in radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
                code < 0: the last_used_position/last_used_tcp_speed/last_used_tcp_acc will not be modified
                code >= 0: the last_used_position/last_used_tcp_speed/last_used_tcp_acc will be modified
        """
        return self._arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, radius=radius,
                                      speed=speed, mvacc=mvacc, mvtime=mvtime, relative=relative,
                                      is_radian=is_radian, wait=wait, timeout=timeout, **kwargs)

    def set_tool_position(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0,
                          speed=None, mvacc=None, mvtime=None, is_radian=None,
                          wait=False, timeout=None, **kwargs):
        """
        Movement relative to the tool coordinate system
        Note:
            1. This interface is moving relative to the current tool coordinate system
            2. The tool coordinate system is not fixed and varies with position.
            3. This interface is only used in the tool coordinate system.


        :param x: the x coordinate relative to the current tool coordinate system, (unit: mm), default is 0
        :param y: the y coordinate relative to the current tool coordinate system, (unit: mm), default is 0
        :param z: the z coordinate relative to the current tool coordinate system, (unit: mm), default is 0
        :param roll: the rotate around the X axis relative to the current tool coordinate system, (unit: rad if is_radian is True else °), default is 0
        :param pitch: the rotate around the Y axis relative to the current tool coordinate system, (unit: rad if is_radian is True else °), default is 0
        :param yaw: the rotate around the Z axis relative to the current tool coordinate system, (unit: rad if is_radian is True else °), default is 0
        :param speed: move speed (mm/s, rad/s), default is self.last_used_tcp_speed
        :param mvacc: move acceleration (mm/s^2, rad/s^2), default is self.last_used_tcp_acc
        :param mvtime: 0, reserved
        :param is_radian: the roll/pitch/yaw in radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
                code < 0: the last_used_tcp_speed/last_used_tcp_acc will not be modified
                code >= 0: the last_used_tcp_speed/last_used_tcp_acc will be modified
        """
        return self._arm.set_tool_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                           speed=speed, mvacc=mvacc, mvtime=mvtime,
                                           is_radian=is_radian, wait=wait, timeout=timeout, **kwargs)

    def get_servo_angle(self, servo_id=None, is_radian=None):
        """
        Get the servo angle
        Note:
            1. If the value you want to return is an radian unit, please set the parameter is_radian to True
                ex: code, angles = arm.get_servo_angle(is_radian=True)
            2. If you want to return only the angle of a single joint, please set the parameter servo_id
                ex: code, angle = arm.get_servo_angle(servo_id=2)
            3. This interface is only used in the base coordinate system.

        :param servo_id: 1-(Number of axes), None(8), default is None
        :param is_radian: the returned value is in radians or not, default is self.default_is_radian
        :return: tuple((code, angle list if servo_id is None or 8 else angle)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_servo_angle(servo_id=servo_id, is_radian=is_radian)

    def set_servo_angle(self, servo_id=None, angle=None, speed=None, mvacc=None, mvtime=None,
                        relative=False, is_radian=None, wait=False, timeout=None, radius=None, **kwargs):
        """
        Set the servo angle, the API will modify self.last_used_angles value
        Note:
            1. If the parameter angle you are passing is an radian unit, be sure to set the parameter is_radian to True.
                ex: code = arm.set_servo_angle(servo_id=1, angle=1.57, is_radian=True)
            2. If you want to wait for the rbt_s to complete this action and then return, please set the parameter wait to True.
                ex: code = arm.set_servo_angle(servo_id=1, angle=45, is_radian=False,wait=True)
            3. This interface is only used in the base coordinate system.

        :param servo_id: 1-(Number of axes), None(8)
            1. 1-(Number of axes) indicates the corresponding joint, the parameter angle should be a numeric value
                ex: code = arm.set_servo_angle(servo_id=1, angle=45, is_radian=False)
            2. None(8) means all joints, default is None, the parameter angle should be a list of values whose length is the number of joints
                ex: code = arm.set_servo_angle(angle=[30, -45, 0, 0, 0, 0, 0], is_radian=False)
        :param angle: angle or angle list, (unit: rad if is_radian is True else °)
            1. If servo_id is 1-(Number of axes), angle should be a numeric value
                ex: code = arm.set_servo_angle(servo_id=1, angle=45, is_radian=False)
            2. If servo_id is None or 8, angle should be a list of values whose length is the number of joints
                like [axis-1, axis-2, axis-3, axis-3, axis-4, axis-5, axis-6, axis-7]
                ex: code = arm.set_servo_angle(angle=[30, -45, 0, 0, 0, 0, 0], is_radian=False)
        :param speed: move speed (unit: rad/s if is_radian is True else °/s), default is self.last_used_joint_speed
        :param mvacc: move acceleration (unit: rad/s^2 if is_radian is True else °/s^2), default is self.last_used_joint_acc
        :param mvtime: 0, reserved
        :param relative: relative move or not
        :param is_radian: the angle in radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :param radius: move radius, if radius is None or radius less than 0, will MoveJoint, else MoveArcJoint
            Note: Only available if version > 1.5.20
            Note: The blending radius cannot be greater than the track length.
            MoveJoint: joint motion
                ex: code = arm.set_servo_angle(..., radius=None)
            MoveArcJoint: joint fusion motion with interpolation
                ex: code = arm.set_servo_angle(..., radius=0)
                Note: Need to set radius>=0
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
                code < 0: the last_used_angles/last_used_joint_speed/last_used_joint_acc will not be modified
                code >= 0: the last_used_angles/last_used_joint_speed/last_used_joint_acc will be modified
        """
        return self._arm.set_servo_angle(servo_id=servo_id, angle=angle, speed=speed, mvacc=mvacc, mvtime=mvtime,
                                         relative=relative, is_radian=is_radian, wait=wait, timeout=timeout, radius=radius, **kwargs)

    def set_servo_angle_j(self, angles, speed=None, mvacc=None, mvtime=None, is_radian=None, **kwargs):
        """
        Set the servo angle, execute only the last instruction, need to be set to servo motion mode(self.set_mode(1))
        Note:
            1. This interface does not modify the value of last_used_angles/last_used_joint_speed/last_used_joint_acc
            2. This interface is only used in the base coordinate system.

        :param angles: angle list, (unit: rad if is_radian is True else °)
        :param speed: speed, reserved
        :param mvacc: acceleration, reserved
        :param mvtime: 0, reserved
        :param is_radian: the angles in radians or not, default is self.default_is_radian
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_servo_angle_j(angles, speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian, **kwargs)

    def set_servo_cartesian(self, mvpose, speed=None, mvacc=None, mvtime=0, is_radian=None, is_tool_coord=False, **kwargs):
        """
        Set the servo cartesian, execute only the last instruction, need to be set to servo motion mode(self.set_mode(1))

        :param mvpose: cartesian position, [x(mm), y(mm), z(mm), roll(rad or °), pitch(rad or °), yaw(rad or °)]
        :param speed: move speed (mm/s), reserved
        :param mvacc: move acceleration (mm/s^2), reserved
        :param mvtime: 0, reserved
        :param is_radian: the roll/pitch/yaw of mvpose in radians or not, default is self.default_is_radian
        :param is_tool_coord: is tool coordinate or not
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_servo_cartesian(mvpose, speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian,
                                             is_tool_coord=is_tool_coord, **kwargs)

    def move_circle(self, pose1, pose2, percent, speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None, **kwargs):
        """
        The motion calculates the trajectory of the space circle according to the three-point coordinates.
        The three-point coordinates are (current starting point, pose1, pose2).

        :param pose1: cartesian position, [x(mm), y(mm), z(mm), roll(rad or °), pitch(rad or °), yaw(rad or °)]
        :param pose2: cartesian position, [x(mm), y(mm), z(mm), roll(rad or °), pitch(rad or °), yaw(rad or °)]
        :param percent: the percentage of arc length and circumference of the movement
        :param speed: move speed (mm/s, rad/s), default is self.last_used_tcp_speed
        :param mvacc: move acceleration (mm/s^2, rad/s^2), default is self.last_used_tcp_acc
        :param mvtime: 0, reserved
        :param is_radian: roll/pitch/yaw value is radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :param kwargs: reserved
        :return: code
            code: See the API code documentation for details.
                code < 0: the last_used_tcp_speed/last_used_tcp_acc will not be modified
                code >= 0: the last_used_tcp_speed/last_used_tcp_acc will be modified
        """
        return self._arm.move_circle(pose1, pose2, percent, speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian, wait=wait, timeout=timeout, **kwargs)

    def move_gohome(self, speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None):
        """
        Move to go home (Back to zero), the API will modify self.last_used_position and self.last_used_angles value
        Warnning: without limit detection
        Note:
            1. The API will change self.last_used_position value into [201.5, 0, 140.5, -180, 0, 0]
            2. The API will change self.last_used_angles value into [0, 0, 0, 0, 0, 0, 0]
            3. If you want to wait for the rbt_s to complete this action and then return, please set the parameter wait to True.
                ex: code = arm.move_gohome(wait=True)
            4. This interface does not modify the value of last_used_angles/last_used_joint_speed/last_used_joint_acc

        :param speed: gohome speed (unit: rad/s if is_radian is True else °/s), default is 50 °/s
        :param mvacc: gohome acceleration (unit: rad/s^2 if is_radian is True else °/s^2), default is 5000 °/s^2
        :param mvtime: reserved
        :param is_radian: the speed and acceleration are in radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.move_gohome(speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian, wait=wait, timeout=timeout)

    def move_arc_lines(self, paths, is_radian=None, times=1, first_pause_time=0.1, repeat_pause_time=0,
                       automatic_calibration=True, speed=None, mvacc=None, mvtime=None, wait=False):
        """
        Continuous linear motion with interpolation.
        Note:
            1. If an error occurs, it will return early.
            2. If the emergency_stop interface is called actively, it will return early.
            3. The last_used_position/last_used_tcp_speed/last_used_tcp_acc will be modified.
            4. The last_used_angles/last_used_joint_speed/last_used_joint_acc will not be modified.

        :param paths: cartesian path list
            1. Specify arc radius:  [[x, y, z, roll, pitch, yaw, radius], ....]
            2. Do not specify arc radius (radius=0):  [[x, y, z, roll, pitch, yaw], ....]
            3. If you want to plan the continuous motion,set radius>0.

        :param is_radian: roll/pitch/yaw of paths are in radians or not, default is self.default_is_radian
        :param times: repeat times, 0 is infinite loop, default is 1
        :param first_pause_time: sleep time at first, purpose is to cache the commands and plan continuous motion, default is 0.1s
        :param repeat_pause_time: interval between repeated movements, unit: (s)second
        :param automatic_calibration: automatic calibration or not, default is True
        :param speed: move speed (mm/s, rad/s), default is self.last_used_tcp_speed
        :param mvacc: move acceleration (mm/s^2, rad/s^2), default is self.last_used_tcp_acc
        :param mvtime: 0, reserved
        :param wait: whether to wait for the arm to complete, default is False
        """
        return self._arm.move_arc_lines(paths, is_radian=is_radian, times=times, first_pause_time=first_pause_time,
                                        repeat_pause_time=repeat_pause_time, automatic_calibration=automatic_calibration,
                                        speed=speed, mvacc=mvacc, mvtime=mvtime, wait=wait)

    def set_servo_attach(self, servo_id=None):
        """
        Attach the servo

        :param servo_id: 1-(Number of axes), 8, if servo_id is 8, will attach all servo
            1. 1-(Number of axes): attach only one joint
                ex: arm.set_servo_attach(servo_id=1)
            2: 8: attach all joints
                ex: arm.set_servo_attach(servo_id=8)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_servo_attach(servo_id=servo_id)

    def set_servo_detach(self, servo_id=None):
        """
        Detach the servo, be sure to do protective work before unlocking to avoid injury or damage.

        :param servo_id: 1-(Number of axes), 8, if servo_id is 8, will detach all servo
            1. 1-(Number of axes): detach only one joint
                ex: arm.set_servo_detach(servo_id=1)
            2: 8: detach all joints, please
                ex: arm.set_servo_detach(servo_id=8)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_servo_detach(servo_id=servo_id)

    def get_version(self):
        """
        Get the xArm firmware version

        :return: tuple((code, version)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_version()

    def get_robot_sn(self):
        """
        Get the xArm sn

        :return: tuple((code, sn)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_robot_sn()

    def check_verification(self):
        """
        check verification

        :return: tuple((code, status)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            status:
                0: verified
                other: not verified
        """
        return self._arm.check_verification()

    def shutdown_system(self, value=1):
        """
        Shutdown the xArm controller system

        :param value: 1: remote shutdown
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.shutdown_system(value=value)

    def get_trajectories(self):
        """
        get the trajectories

        Note:
            1. This interface relies on xArmStudio 1.2.0 or above
            2. This interface relies on Firmware 1.2.0 or above

        :return: tuple((code, trajectories))
            code: See the API code documentation for details.
            trajectories: [{
                'name': name, # The name of the trajectory
                'duration': duration, # The duration of the trajectory (seconds)
            }]
        """
        return self._arm.get_trajectories()

    def start_record_trajectory(self):
        """
        Start trajectory recording, only in teach mode, so you need to set joint teaching mode before.

        Note:
            1. This interface relies on Firmware 1.2.0 or above
            2. set joint teaching mode: set_mode(2);set_state(0)

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.start_record_trajectory()

    def stop_record_trajectory(self, filename=None):
        """
        Stop trajectory recording

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param filename: The name to save
            1. Only strings consisting of English or numbers are supported, and the length is no more than 50.
            2. The trajectory is saved in the controller box.
            3. If the filename is None, just stop recording, do not save, you need to manually call `save_record_trajectory` save before changing the mode. otherwise it will be lost
            4. This action will overwrite the trajectory with the same name
            5. Empty the trajectory in memory after saving
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.stop_record_trajectory(filename=filename)

    def save_record_trajectory(self, filename, wait=True, timeout=2):
        """
        Save the trajectory you just recorded

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param filename: The name to save
            1. Only strings consisting of English or numbers are supported, and the length is no more than 50.
            2. The trajectory is saved in the controller box.
            3. This action will overwrite the trajectory with the same name
            4. Empty the trajectory in memory after saving, so repeated calls will cause the recorded trajectory to be covered by an empty trajectory.
        :param wait: Whether to wait for saving, default is True
        :param timeout: Timeout waiting for saving to complete
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.save_record_trajectory(filename, wait=wait, timeout=timeout)

    def load_trajectory(self, filename, wait=True, timeout=10):
        """
        Load the trajectory

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param filename: The name of the trajectory to load
        :param wait: Whether to wait for loading, default is True
        :param timeout: Timeout waiting for loading to complete
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.load_trajectory(filename, wait=wait, timeout=timeout)

    def playback_trajectory(self, times=1, filename=None, wait=True, double_speed=1):
        """
        Playback trajectory

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param times: Number of playbacks,
            1. Only valid when the current position of the arm is the end position of the track, otherwise it will only be played once.
        :param filename: The name of the trajectory to play back
            1. If filename is None, you need to manually call the `load_trajectory` to load the trajectory.
        :param wait: whether to wait for the arm to complete, default is False
        :param double_speed: double speed, only support 1/2/4, default is 1, only available if version > 1.2.11
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.playback_trajectory(times=times, filename=filename, wait=wait, double_speed=double_speed)

    def get_trajectory_rw_status(self):
        """
        Get trajectory read/write status

        :return: (code, status)
            code: See the API code documentation for details.
            status:
                0: no read/write
                1: loading
                2: load success
                3: load failed
                4: saving
                5: save success
                6: save failed
        """
        return self._arm.get_trajectory_rw_status()

    def get_reduced_mode(self):
        """
        Get reduced mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :return: tuple((code, mode))
            code: See the API code documentation for details.
            mode: 0 or 1, 1 means that the reduced mode is turned on. 0 means that the reduced mode is not turned on
        """
        return self._arm.get_reduced_mode()

    def get_reduced_states(self, is_radian=None):
        """
        Get states of the reduced mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param is_radian: the max_joint_speed of the states is in radians or not, default is self.default_is_radian
        :return: tuple((code, states))
            code: See the API code documentation for details.
            states: [....]
                if version > 1.2.11:
                    states: [
                        reduced_mode_is_on,
                        [reduced_x_max, reduced_x_min, reduced_y_max, reduced_y_min, reduced_z_max, reduced_z_min],
                        reduced_max_tcp_speed,
                        reduced_max_joint_speed,
                        joint_ranges([joint-1-min, joint-1-max, ..., joint-7-min, joint-7-max]),
                        safety_boundary_is_on,
                        collision_rebound_is_on,
                    ]`
                if version <= 1.2.11:
                    states: [
                        reduced_mode_is_on,
                        [reduced_x_max, reduced_x_min, reduced_y_max, reduced_y_min, reduced_z_max, reduced_z_min],
                        reduced_max_tcp_speed,
                        reduced_max_joint_speed,
                    ]`
        """
        return self._arm.get_reduced_states(is_radian=is_radian)

    def set_reduced_mode(self, on):
        """
        Turn on/off reduced mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above

        :param on: True/False
                   such as:Turn on the reduced mode : code=arm.set_reduced_mode(True)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_reduced_mode(on)

    def set_reduced_max_tcp_speed(self, speed):
        """
        Set the maximum tcp speed of the reduced mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above
            2. Only reset the reduced mode to take effect (`set_reduced_mode(True)`)

        :param speed: speed (mm/s)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_reduced_max_tcp_speed(speed)

    def set_reduced_max_joint_speed(self, speed, is_radian=None):
        """
        Set the maximum joint speed of the reduced mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above
            2. Only reset the reduced mode to take effect (`set_reduced_mode(True)`)

        :param speed: speed (°/s or rad/s)
        :param is_radian: the speed is in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_reduced_max_joint_speed(speed, is_radian=is_radian)

    def set_reduced_tcp_boundary(self, boundary):
        """
        Set the boundary of the safety boundary mode

        Note:
            1. This interface relies on Firmware 1.2.0 or above
            2. Only reset the reduced mode to take effect (`set_reduced_mode(True)`)

        :param boundary: [x_max, x_min, y_max, y_min, z_max, z_min]
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_reduced_tcp_boundary(boundary)

    def set_reduced_joint_range(self, joint_range, is_radian=None):
        """
        Set the joint range of the reduced mode

        Note:
            1. This interface relies on Firmware 1.2.11 or above
            2. Only reset the reduced mode to take effect (`set_reduced_mode(True)`)

        :param joint_range: [joint-1-min, joint-1-max, ..., joint-7-min, joint-7-max]
        :param is_radian: the param joint_range are in radians or not, default is self.default_is_radian
        :return:
        """
        return self._arm.set_reduced_joint_range(joint_range, is_radian=is_radian)

    def set_fence_mode(self, on):
        """
        Set the fence mode,turn on/off fense mode

        Note:
            1. This interface relies on Firmware 1.2.11 or above

        :param on: True/False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_fense_mode(on)

    def set_collision_rebound(self, on):
        """
        Set the collision rebound,turn on/off collision rebound

        Note:
            1. This interface relies on Firmware 1.2.11 or above

        :param on: True/False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_collision_rebound(on)

    def set_world_offset(self, offset, is_radian=None):
        """
        Set the base coordinate offset

        Note:
            1. This interface relies on Firmware 1.2.11 or above

        :param offset: [x, y, z, roll, pitch, yaw]
        :param is_radian: the roll/pitch/yaw in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_world_offset(offset, is_radian=is_radian)

    def get_is_moving(self):
        """
        Check xArm is moving or not
        :return: True/False
        """
        return self._arm.get_is_moving()

    def get_state(self):
        """
        Get state

        :return: tuple((code, state)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            state:
                1: in motion
                2: sleeping
                3: suspended
                4: stopping
        """
        return self._arm.get_state()

    def set_state(self, state=0):
        """
        Set the xArm state

        :param state: default is 0
            0: sport state
            3: pause state
            4: stop state
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_state(state=state)

    def set_mode(self, mode=0):
        """
        Set the xArm mode

        :param mode: default is 0
            0: position control mode
            1: servo motion mode
                Note: the use of the set_servo_angle_j interface must first be set to this mode
                Note: the use of the set_servo_cartesian interface must first be set to this mode
            2: joint teaching mode
                Note: use this mode to ensure that the arm has been identified and the control box and arm used for identification are one-to-one.
            3: cartesian teaching mode (invalid)
            4: joint velocity control mode
            5: cartesian velocity control mode
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_mode(mode=mode)

    def get_cmdnum(self):
        """
        Get the cmd count in cache
        :return: tuple((code, cmd_num)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_cmdnum()

    def get_err_warn_code(self, show=False, lang='en'):
        """
        Get the controller error and warn code

        :param show: show the detail info if True
        :param lang: show language, en/cn, degault is en, only available if show is True
        :return: tuple((code, [error_code, warn_code])), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            error_code: See Chapter 7 of the xArm User Manual for details.
            warn_code: See Chapter 7 of the xArm User Manual for details.
        """
        return self._arm.get_err_warn_code(show=show, lang=lang)

    def clean_error(self):
        """
        Clean the error, need to be manually enabled motion(arm.motion_enable(True)) and set state(arm.set_state(state=0))after clean error

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.clean_error()

    def clean_warn(self):
        """
        Clean the warn

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.clean_warn()

    def motion_enable(self, enable=True, servo_id=None):
        """
        Motion enable

        :param enable:True/False
        :param servo_id: 1-(Number of axes), None(8)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.motion_enable(servo_id=servo_id, enable=enable)

    def reset(self, speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None):
        """
        Reset the xArm
        Warnning: without limit detection
        Note:
            1. If there are errors or warnings, this interface will clear the warnings and errors.
            2. If not ready, the api will auto enable motion and set state
            3. This interface does not modify the value of last_used_angles/last_used_joint_speed/last_used_joint_acc

        :param speed: reset speed (unit: rad/s if is_radian is True else °/s), default is 50 °/s
        :param mvacc: reset acceleration (unit: rad/s^2 if is_radian is True else °/s^2), default is 5000 °/s^2
        :param mvtime: reserved
        :param is_radian: the speed and acceleration are in radians or not, default is self.default_is_radian
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        """
        return self._arm.reset(speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian, wait=wait, timeout=timeout)

    def set_pause_time(self, sltime, wait=False):
        """
        Set the arm pause time, xArm will pause sltime second

        :param sltime: sleep time,unit:(s)second
        :param wait: wait or not, default is False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_pause_time(sltime, wait=wait)

    def set_tcp_offset(self, offset, is_radian=None, **kwargs):
        """
        Set the tool coordinate system offset at the end
        Note:
            1. Do not use if not required
            2. If not saved and you want to revert to the last saved value, please reset the offset by set_tcp_offset([0, 0, 0, 0, 0, 0])
            3. If not saved, it will be lost after reboot
            4. The save_conf interface can record the current settings and will not be lost after the restart.
            5. The clean_conf interface can restore system default settings

        :param offset: [x, y, z, roll, pitch, yaw]
        :param is_radian: the roll/pitch/yaw in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tcp_offset(offset, is_radian=is_radian, **kwargs)

    def set_tcp_jerk(self, jerk):
        """
        Set the translational jerk of Cartesian space
        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param jerk: jerk (mm/s^3)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tcp_jerk(jerk)

    def set_tcp_maxacc(self, acc):
        """
        Set the max translational acceleration of Cartesian space
        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param acc: max acceleration (mm/s^2)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tcp_maxacc(acc)

    def set_joint_jerk(self, jerk, is_radian=None):
        """
        Set the jerk of Joint space
        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param jerk: jerk (°/s^3 or rad/s^3)
        :param is_radian: the jerk in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_joint_jerk(jerk, is_radian=is_radian)

    def set_joint_maxacc(self, acc, is_radian=None):
        """
        Set the max acceleration of Joint space

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param acc: max acceleration (°/s^2 or rad/s^2)
        :param is_radian: the jerk in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_joint_maxacc(acc, is_radian=is_radian)

    def set_tcp_load(self, weight, center_of_gravity):
        """
        Set the end load of xArm

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param weight: load weight (unit: kg)
        :param center_of_gravity: load center of gravity, such as [x(mm), y(mm), z(mm)]
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tcp_load(weight, center_of_gravity)

    def set_collision_sensitivity(self, value):
        """
        Set the sensitivity of collision

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param value: sensitivity value, 0~5
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_collision_sensitivity(value)

    def set_teach_sensitivity(self, value):
        """
        Set the sensitivity of drag and teach

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param value: sensitivity value, 1~5
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_teach_sensitivity(value)

    def set_gravity_direction(self, direction):
        """
        Set the direction of gravity

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param direction: direction of gravity, such as [x(mm), y(mm), z(mm)]
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_gravity_direction(direction=direction)

    def set_mount_direction(self, base_tilt_deg, rotation_deg, is_radian=None):
        """
        Set the mount direction

        Note:
            1. Do not use if not required
            2. If not saved, it will be lost after reboot
            3. The save_conf interface can record the current settings and will not be lost after the restart.
            4. The clean_conf interface can restore system default settings

        :param base_tilt_deg: tilt degree
        :param rotation_deg: rotation degree
        :param is_radian: the base_tilt_deg/rotation_deg in radians or not, default is self.default_is_radian
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_mount_direction(base_tilt_deg, rotation_deg, is_radian=is_radian)

    def clean_conf(self):
        """
        Clean current config and restore system default settings
        Note:
            1. This interface will clear the current settings and restore to the original settings (system default settings)

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.clean_conf()

    def save_conf(self):
        """
        Save config
        Note:
            1. This interface can record the current settings and will not be lost after the restart.
            2. The clean_conf interface can restore system default settings

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.save_conf()

    def get_inverse_kinematics(self, pose, input_is_radian=None, return_is_radian=None):
        """
        Get inverse kinematics

        :param pose: [x(mm), y(mm), z(mm), roll(rad or °), pitch(rad or °), yaw(rad or °)]
            Note: the roll/pitch/yaw unit is radian if input_is_radian is True, else °
        :param input_is_radian: the param pose value(only roll/pitch/yaw) is in radians or not, default is self.default_is_radian
        :param return_is_radian: the returned value is in radians or not, default is self.default_is_radian
        :return: tuple((code, angles)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            angles: [angle-1(rad or °), angle-2, ..., angle-(Number of axes)] or []
                Note: the returned angle value is radians if return_is_radian is True, else °
        """
        return self._arm.get_inverse_kinematics(pose, input_is_radian=input_is_radian, return_is_radian=return_is_radian)

    def get_forward_kinematics(self, angles, input_is_radian=None, return_is_radian=None):
        """
        Get forward kinematics

        :param angles: [angle-1, angle-2, ..., angle-n], n is the number of axes of the arm
        :param input_is_radian: the param angles value is in radians or not, default is self.default_is_radian
        :param return_is_radian: the returned value is in radians or not, default is self.default_is_radian
        :return: tuple((code, pose)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            pose: [x(mm), y(mm), z(mm), roll(rad or °), pitch(rad or °), yaw(rad or °)] or []
                Note: the roll/pitch/yaw value is radians if return_is_radian is True, else °
        """
        return self._arm.get_forward_kinematics(angles, input_is_radian=input_is_radian, return_is_radian=return_is_radian)

    def is_tcp_limit(self, pose, is_radian=None):
        """
        Check the tcp pose is in limit

        :param pose: [x, y, z, roll, pitch, yaw]
        :param is_radian: roll/pitch/yaw value is radians or not, default is self.default_is_radian
        :return: tuple((code, limit)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            limit: True/False/None, limit or not, or failed
        """
        return self._arm.is_tcp_limit(pose, is_radian=is_radian)

    def is_joint_limit(self, joint, is_radian=None):
        """
        Check the joint angle is in limit

        :param joint: [angle-1, angle-2, ..., angle-n], n is the number of axes of the arm
        :param is_radian: angle value is radians or not, default is self.default_is_radian
        :return: tuple((code, limit)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            limit: True/False/None, limit or not, or failed
        """
        return self._arm.is_joint_limit(joint, is_radian=is_radian)

    def emergency_stop(self):
        """
        Emergency stop (set_state(4) -> motion_enable(True) -> set_state(0))
        Note:
            1. This interface does not automatically clear the error. If there is an error, you need to handle it according to the error code.
        """
        return self._arm.emergency_stop()

    def set_gripper_enable(self, enable, **kwargs):
        """
        Set the gripper enable

        :param enable: enable or not
         Note:  such as code = arm.set_gripper_enable(True)  #turn on the Gripper
        :return: code
            code: See the Gripper code documentation for details.
        """
        return self._arm.set_gripper_enable(enable, **kwargs)

    def set_gripper_mode(self, mode, **kwargs):
        """
        Set the gripper mode

        :param mode: 0: location mode
         Note:  such as code = rm.set_gripper_mode(0)
        :return: code
            code: See the Gripper code documentation for details.
        """
        return self._arm.set_gripper_mode(mode, **kwargs)

    def get_gripper_position(self, **kwargs):
        """
        Get the gripper position

        :return: tuple((code, pos)), only when code is 0, the returned result is correct.
            code: See the Gripper code documentation for details.
        """
        return self._arm.get_gripper_position(**kwargs)

    def set_gripper_position(self, pos, wait=False, speed=None, auto_enable=False, timeout=None, **kwargs):
        """
        Set the gripper position

        :param pos: pos
        :param wait: wait or not, default is False
        :param speed: speed,unit:r/min
        :param auto_enable: auto enable or not, default is False
        :param timeout: wait time, unit:second, default is 10s
        :return: code
            code: See the Gripper code documentation for details.
        """
        return self._arm.set_gripper_position(pos, wait=wait, speed=speed, auto_enable=auto_enable, timeout=timeout, **kwargs)

    def set_gripper_speed(self, speed, **kwargs):
        """
        Set the gripper speed

        :param speed:
        :return: code
            code: See the Gripper code documentation for details.
        """
        return self._arm.set_gripper_speed(speed, **kwargs)

    def get_gripper_err_code(self, **kwargs):
        """
        Get the gripper error code

        :return: tuple((code, err_code)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            err_code: See the Gripper code documentation for details.
        """
        return self._arm.get_gripper_err_code(**kwargs)

    def clean_gripper_error(self, **kwargs):
        """
        Clean the gripper error

        :return: code
            code: See the Gripper code documentation for details.
        """
        return self._arm.clean_gripper_error(**kwargs)

    def get_tgpio_digital(self, ionum=None):
        """
        Get the digital value of the specified Tool GPIO

        :param ionum: 0 or 1 or None(both 0 and 1), default is None
        :return: tuple((code, value or value list)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_tgpio_digital(ionum)

    def set_tgpio_digital(self, ionum, value, delay_sec=None):
        """
        Set the digital value of the specified Tool GPIO
        
        :param ionum: 0 or 1
        :param value: value
        :param delay_sec: delay effective time from the current start, in seconds, default is None(effective immediately)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tgpio_digital(ionum=ionum, value=value, delay_sec=delay_sec)

    def get_tgpio_analog(self, ionum=None):
        """
        Get the analog value of the specified Tool GPIO
        :param ionum: 0 or 1 or None(both 0 and 1), default is None
        :return: tuple((code, value or value list)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_tgpio_analog(ionum)

    def get_vacuum_gripper(self):
        """
        Get vacuum gripper state

        :return: tuple((code, state)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            state: suction cup state
                0: suction cup is off
                1: suction cup is on
        """
        return self._arm.get_suction_cup()

    def set_vacuum_gripper(self, on, wait=False, timeout=3, delay_sec=None):
        """
        Set vacuum gripper state

        :param on: open or not
            on=True: equivalent to calling `set_tgpio_digital(0, 1)` and `set_tgpio_digital(1, 0)`
            on=False: equivalent to calling `set_tgpio_digital(0, 0)` and `set_tgpio_digital(1, 1)`
        :param wait: wait or not, default is False
        :param timeout: wait time, unit:second, default is 3s
        :param delay_sec: delay effective time from the current start, in seconds, default is None(effective immediately)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_suction_cup(on, wait=wait, timeout=timeout, delay_sec=delay_sec)

    def get_cgpio_digital(self, ionum=None):
        """
        Get the digital value of the specified Controller GPIO

        :param ionum: 0~7 or None(both 0~7), default is None
        :return: tuple((code, value or value list)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_cgpio_digital(ionum=ionum)

    def get_cgpio_analog(self, ionum=None):
        """
        Get the analog value of the specified Controller GPIO
        :param ionum: 0 or 1 or None(both 0 and 1), default is None
        :return: tuple((code, value or value list)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_cgpio_analog(ionum=ionum)

    def set_cgpio_digital(self, ionum, value, delay_sec=None):
        """
        Set the digital value of the specified Controller GPIO

        :param ionum: 0~7
        :param value: value
        :param delay_sec: delay effective time from the current start, in seconds, default is None(effective immediately)
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_cgpio_digital(ionum=ionum, value=value, delay_sec=delay_sec)

    def set_cgpio_analog(self, ionum, value):
        """
        Set the analog value of the specified Controller GPIO

        :param ionum: 0 or 1
        :param value: value
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_cgpio_analog(ionum=ionum, value=value)

    def set_cgpio_digital_input_function(self, ionum, fun):
        """
        Set the digital input functional mode of the Controller GPIO
        :param ionum: 0~7
        :param fun: functional mode
            0: general input
            1: external emergency stop
            2: reversed, protection reset
            3: reversed, reduced mode
            4: reversed, operating mode
            5: reversed, three-state switching signal
            11: offline task
            12: teaching mode
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_cgpio_digital_input_function(ionum=ionum, fun=fun)

    def set_cgpio_digital_output_function(self, ionum, fun):
        """
        Set the digital output functional mode of the specified Controller GPIO
        :param ionum: 0~7
        :param fun: functionnal mode
            0: general output
            1: emergency stop
            2: in motion
            11: has error
            12: has warn
            13: in collision
            14: in teaching
            15: in offline task
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_cgpio_digital_output_function(ionum=ionum, fun=fun)

    def get_cgpio_state(self):
        """
        Get the state of the Controller GPIO
        :return: code, states
            code: See the API code documentation for details.
            states: [...]
                states[0]: contorller gpio module state
                    states[0] == 0: normal
                    states[0] == 1: wrong
                    states[0] == 6: communication failure
                states[1]: controller gpio module error code
                    states[1] == 0: normal
                    states[1] != 0: error code
                states[2]: digital input functional gpio state
                    Note: digital-i-input functional gpio state = states[2] >> i & 0x01
                states[3]: digital input configuring gpio state
                    Note: digital-i-input configuring gpio state = states[3] >> i & 0x01
                states[4]: digital output functional gpio state
                    Note: digital-i-output functional gpio state = states[4] >> i & 0x01
                states[5]: digital output configuring gpio state
                    Note: digital-i-output configuring gpio state = states[5] >> i & 0x01
                states[6]: analog-0 input value
                states[7]: analog-1 input value
                states[8]: analog-0 output value
                states[9]: analog-1 output value
                states[10]: digital input functional info, [digital-0-input-functional-mode, ... digital-7-input-functional-mode]
                states[11]: digital output functional info, [digital-0-output-functional-mode, ... digital-7-output-functional-mode]
        """
        return self._arm.get_cgpio_state()

    def register_report_callback(self, callback=None, report_cartesian=True, report_joints=True,
                                 report_state=True, report_error_code=True, report_warn_code=True,
                                 report_mtable=True, report_mtbrake=True, report_cmd_num=True):
        """
        Register the report callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                'cartesian': [], # if report_cartesian is True
                'joints': [], # if report_joints is True
                'error_code': 0, # if report_error_code is True
                'warn_code': 0, # if report_warn_code is True
                'state': state, # if report_state is True
                'mtbrake': mtbrake, # if report_mtbrake is True, and available if enable_report is True and the connect way is socket
                'mtable': mtable, # if report_mtable is True, and available if enable_report is True and the connect way is socket
                'cmdnum': cmdnum, # if report_cmd_num is True
            }
        :param report_cartesian: report cartesian or not, default is True
        :param report_joints: report joints or not, default is True
        :param report_state: report state or not, default is True
        :param report_error_code: report error or not, default is True
        :param report_warn_code: report warn or not, default is True
        :param report_mtable: report motor enable states or not, default is True
        :param report_mtbrake: report motor brake states or not, default is True
        :param report_cmd_num: report cmdnum or not, default is True
        :return: True/False
        """
        return self._arm.register_report_callback(callback=callback,
                                                  report_cartesian=report_cartesian,
                                                  report_joints=report_joints,
                                                  report_state=report_state,
                                                  report_error_code=report_error_code,
                                                  report_warn_code=report_warn_code,
                                                  report_mtable=report_mtable,
                                                  report_mtbrake=report_mtbrake,
                                                  report_cmd_num=report_cmd_num)

    def register_report_location_callback(self, callback=None, report_cartesian=True, report_joints=True):
        """
        Register the report location callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "cartesian": [x, y, z, roll, pitch, yaw], ## if report_cartesian is True
                "joints": [angle-1, angle-2, angle-3, angle-4, angle-5, angle-6, angle-7], ## if report_joints is True
            }
        :param report_cartesian: report or not, True/False, default is True
        :param report_joints: report or not, True/False, default is True
        :return: True/False
        """
        return self._arm.register_report_location_callback(callback=callback,
                                                           report_cartesian=report_cartesian,
                                                           report_joints=report_joints)

    def register_connect_changed_callback(self, callback=None):
        """
        Register the connect status changed callback

        :param callback:
            callback data:
            {
                "connected": connected,
                "reported": reported,
            }
        :return: True/False
        """
        return self._arm.register_connect_changed_callback(callback=callback)

    def register_state_changed_callback(self, callback=None):
        """
        Register the state status changed callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "state": state,
            }
        :return: True/False
        """
        return self._arm.register_state_changed_callback(callback=callback)

    def register_mode_changed_callback(self, callback=None):
        """
        Register the mode changed callback, only available if enable_report is True and the connect way is socket

        :param callback:
            callback data:
            {
                "mode": mode,
            }
        :return: True/False
        """
        return self._arm.register_mode_changed_callback(callback=callback)

    def register_mtable_mtbrake_changed_callback(self, callback=None):
        """
        Register the motor enable states or motor brake states changed callback, only available if enable_report is True and the connect way is socket

        :param callback:
            callback data:
            {
                "mtable": [motor-1-motion-enable, motor-2-motion-enable, ...],
                "mtbrake": [motor-1-brake-enable, motor-1-brake-enable,...],
            }
        :return: True/False
        """
        return self._arm.register_mtable_mtbrake_changed_callback(callback=callback)

    def register_error_warn_changed_callback(self, callback=None):
        """
        Register the error code or warn code changed callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "error_code": error_code,
                "warn_code": warn_code,
            }
        :return: True/False
        """
        return self._arm.register_error_warn_changed_callback(callback=callback)

    def register_cmdnum_changed_callback(self, callback=None):
        """
        Register the cmdnum changed callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "cmdnum": cmdnum
            }
        :return: True/False
        """
        return self._arm.register_cmdnum_changed_callback(callback=callback)

    def register_temperature_changed_callback(self, callback=None):
        """
        Register the temperature changed callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "temperatures": [servo-1-temperature, ...., servo-7-temperature]
            }
        :return: True/False
        """
        return self._arm.register_temperature_changed_callback(callback=callback)

    def register_count_changed_callback(self, callback=None):
        """
        Register the counter value changed callback, only available if enable_report is True

        :param callback:
            callback data:
            {
                "count": counter value
            }
        :return: True/False
        """
        return self._arm.register_count_changed_callback(callback=callback)

    def release_report_callback(self, callback=None):
        """
        Release the report callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_report_callback(callback)

    def release_report_location_callback(self, callback=None):
        """
        Release the location report callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_report_location_callback(callback)

    def release_connect_changed_callback(self, callback=None):
        """
        Release the connect changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_connect_changed_callback(callback)

    def release_state_changed_callback(self, callback=None):
        """
        Release the state changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_state_changed_callback(callback)

    def release_mode_changed_callback(self, callback=None):
        """
        Release the mode changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_mode_changed_callback(callback)

    def release_mtable_mtbrake_changed_callback(self, callback=None):
        """
        Release the motor enable states or motor brake states changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_mtable_mtbrake_changed_callback(callback)

    def release_error_warn_changed_callback(self, callback=None):
        """
        Release the error warn changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_error_warn_changed_callback(callback)

    def release_cmdnum_changed_callback(self, callback=None):
        """
        Release the cmdnum changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_cmdnum_changed_callback(callback)

    def release_temperature_changed_callback(self, callback=None):
        """
        Release the temperature changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_temperature_changed_callback(callback=callback)

    def release_count_changed_callback(self, callback=None):
        """
        Release the counter value changed callback

        :param callback:
        :return: True/False
        """
        return self._arm.release_count_changed_callback(callback=callback)

    def get_servo_debug_msg(self, show=False, lang='en'):
        """
        Get the servo debug msg, used only for debugging

        :param show: show the detail info if True
        :param lang: language, en/cn, default is en
        :return: tuple((code, servo_info_list)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_servo_debug_msg(show=show, lang=lang)

    def run_blockly_app(self, path, **kwargs):
        """
        Run the app generated by xArmStudio software
        :param path: app path
        """
        return self._arm.run_blockly_app(path, **kwargs)

    def run_gcode_file(self, path, **kwargs):
        """
        Run the gcode file
        :param path: gcode file path
        """
        return self._arm.run_gcode_file(path, **kwargs)

    def get_gripper_version(self):
        """
        Get gripper version, only for debug

        :return: (code, version)
            code: See the API code documentation for details.
        """
        return self._arm.get_gripper_version()

    def get_servo_version(self, servo_id=1):
        """
        Get servo version, only for debug

        :param servo_id: servo id(1~7)
        :return: (code, version)
            code: See the API code documentation for details.
        """
        return self._arm.get_servo_version(servo_id=servo_id)

    def get_tgpio_version(self):
        """
        Get tool gpio version, only for debug

        :return: (code, version)
            code: See the API code documentation for details.
        """
        return self._arm.get_tgpio_version()

    def get_harmonic_type(self, servo_id=1):
        """
        Get harmonic type, only for debu

        :return: (code, type)
            code: See the API code documentation for details.
        """
        return self._arm.get_harmonic_type(servo_id=servo_id)

    def get_hd_types(self):
        """
        Get harmonic types, only for debug

        :return: (code, types)
            code: See the API code documentation for details.
        """
        return self._arm.get_hd_types()

    def set_counter_reset(self):
        """
        Reset counter value

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_counter_reset()

    def set_counter_increase(self, val=1):
        """
        Set counter plus value, only support plus 1

        :param val: reversed
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_counter_increase(val)

    def set_tgpio_digital_with_xyz(self, ionum, value, xyz, fault_tolerance_radius):
        """
        Set the digital value of the specified Tool GPIO when the rbt_s has reached the specified xyz position
        
        :param ionum: 0 or 1
        :param value: value
        :param xyz: position xyz, as [x, y, z]
        :param fault_tolerance_radius: fault tolerance radius
        :return: code
            code: See the API code documentation for details. 
        """
        return self._arm.set_tgpio_digital_with_xyz(ionum, value, xyz, fault_tolerance_radius)

    def set_cgpio_digital_with_xyz(self, ionum, value, xyz, fault_tolerance_radius):
        """
        Set the digital value of the specified Controller GPIO when the rbt_s has reached the specified xyz position
        
        :param ionum: 0 ~ 7
        :param value: value
        :param xyz: position xyz, as [x, y, z]
        :param fault_tolerance_radius: fault tolerance radius
        :return: code
            code: See the API code documentation for details.  
        """
        return self._arm.set_cgpio_digital_with_xyz(ionum, value, xyz, fault_tolerance_radius)

    def set_cgpio_analog_with_xyz(self, ionum, value, xyz, fault_tolerance_radius):
        """
        Set the analog value of the specified Controller GPIO when the rbt_s has reached the specified xyz position

        :param ionum: 0 ~ 1
        :param value: value
        :param xyz: position xyz, as [x, y, z]
        :param fault_tolerance_radius: fault tolerance radius
        :return: code
            code: See the API code documentation for details.  
        """
        return self._arm.set_cgpio_analog_with_xyz(ionum, value, xyz, fault_tolerance_radius)

    def config_tgpio_reset_when_stop(self, on_off):
        """
        Config the Tool GPIO reset the digital output when the rbt_s is in stop state
        
        :param on_off: True/False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.config_io_reset_when_stop(1, on_off)

    def config_cgpio_reset_when_stop(self, on_off):
        """
        Config the Controller GPIO reset the digital output when the rbt_s is in stop state

        :param on_off: True/False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.config_io_reset_when_stop(0, on_off)

    def set_position_aa(self, axis_angle_pose, speed=None, mvacc=None, mvtime=None,
                        is_radian=None, is_tool_coord=False, relative=False, wait=False, timeout=None, **kwargs):
        """
        Set the pose represented by the axis angle pose
        
        :param axis_angle_pose: the axis angle pose, [x(mm), y(mm), z(mm), rx(rad or °), ry(rad or °), rz(rad or °)]
        :param speed: move speed (mm/s, rad/s), default is self.last_used_tcp_speed
        :param mvacc: move acceleration (mm/s^2, rad/s^2), default is self.last_used_tcp_acc
        :param mvtime: 0, reserved 
        :param is_radian: the rx/ry/rz of axis_angle_pose in radians or not, default is self.default_is_radian
        :param is_tool_coord: is tool coordinate or not
        :param relative: relative move or not
        :param wait: whether to wait for the arm to complete, default is False
        :param timeout: maximum waiting time(unit: second), default is None(no timeout), only valid if wait is True
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_position_aa(axis_angle_pose, speed=speed, mvacc=mvacc, mvtime=mvtime,
                                         is_radian=is_radian, is_tool_coord=is_tool_coord, relative=relative,
                                         wait=wait, timeout=timeout, **kwargs)

    def set_servo_cartesian_aa(self, axis_angle_pose, speed=None, mvacc=None, is_radian=None, is_tool_coord=False, relative=False, **kwargs):
        """
        Set the servo cartesian represented by the axis angle pose, execute only the last instruction, need to be set to servo motion mode(self.set_mode(1))
        Note:
            1. only available if firmware_version >= 1.4.7

        :param axis_angle_pose: the axis angle pose, [x(mm), y(mm), z(mm), rx(rad or °), ry(rad or °), rz(rad or °)]
        :param speed: move speed (mm/s), reserved
        :param mvacc: move acceleration (mm/s^2), reserved
        :param is_radian: the rx/ry/rz of axis_angle_pose in radians or not, default is self.default_is_radian
        :param is_tool_coord: is tool coordinate or not
        :param relative: relative move or not
        :return: code
            code: See the API code documentation for details.
        """

        return self._arm.set_servo_cartesian_aa(axis_angle_pose, speed=speed, mvacc=mvacc, is_radian=is_radian,
                                                is_tool_coord=is_tool_coord, relative=relative, **kwargs)

    def get_pose_offset(self, pose1, pose2, orient_type_in=0, orient_type_out=0, is_radian=None):
        """
        Calculate the pose offset of two given points
        
        :param pose1: [x(mm), y(mm), z(mm), roll/rx(rad or °), pitch/ry(rad or °), yaw/rz(rad or °)]
        :param pose2: [x(mm), y(mm), z(mm), roll/rx(rad or °), pitch/ry(rad or °), yaw/rz(rad or °)]
        :param orient_type_in: input attitude notation, 0 is RPY(roll/pitch/yaw) (default), 1 is axis angle(rx/ry/rz)
        :param orient_type_out: notation of output attitude, 0 is RPY (default), 1 is axis angle
        :param is_radian: the roll/rx/pitch/ry/yaw/rz of pose1/pose2/return_pose is radian or not
        :return: tuple((code, pose)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            pose: [x(mm), y(mm), z(mm), roll/rx(rad or °), pitch/ry(rad or °), yaw/rz(rad or °)]
        """
        return self._arm.get_pose_offset(pose1, pose2, orient_type_in=orient_type_in, orient_type_out=orient_type_out,
                                         is_radian=is_radian)

    def get_position_aa(self, is_radian=None):
        """
        Get the pose represented by the axis angle pose
        
        :param is_radian: the returned value (only rx/ry/rz) is in radians or not, default is self.default_is_radian
        :return: tuple((code, [x, y, z, rx, ry, rz])), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
        """
        return self._arm.get_position_aa(is_radian=is_radian)

    def get_joints_torque(self):
        """
        Get joints torque
        
        :return: tuple((code, joints_torque))
            code: See the API code documentation for details.
            joints_torque: joints torque
        """
        return self._arm.get_joints_torque()

    def set_joints_torque(self, joints_torque):
        """
        Set joints torque,
            Warning: If necessary, please do not set it randomly, it may damage the rbt_s arm

        :param joints_torque: 
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_joints_torque(joints_torque)

    def get_safe_level(self):
        """
        Get safe level
        
        :return: tuple((code, safe_level))
            code: See the API code documentation for details.
            safe_level: safe level
        """
        return self._arm.get_safe_level()

    def set_safe_level(self, level=4):
        """
        Set safe level,

        :param level: safe level, default is 4
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_safe_level(level=level)

    def set_timeout(self, timeout):
        """
        Set the timeout of cmd response

        :param timeout: seconds
        """
        return self._arm.set_timeout(timeout)

    def robotiq_reset(self):
        """
        Reset the robotiq gripper (clear previous activation if any)
        
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation
        """
        return self._arm.robotiq_reset()

    def robotiq_set_activate(self, wait=True, timeout=3):
        """
        If not already activated. Activate the robotiq gripper
        
        :param wait: whether to wait for the robotiq activate complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 3, only available if wait=True
        
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation 
        """
        return self._arm.robotiq_set_activate(wait=wait, timeout=timeout)

    def robotiq_set_position(self, pos, speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs):
        """
        Go to the position with determined speed and force.
        
        :param pos: position of the gripper. Integer between 0 and 255. 0 being the open position and 255 being the close position.
        :param speed: gripper speed between 0 and 255
        :param force: gripper force between 0 and 255
        :param wait: whether to wait for the robotion motion complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 5, only available if wait=True
        
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation 
        """
        return self._arm.robotiq_set_position(pos, speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)

    def robotiq_open(self, speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs):
        """
        Open the robotiq gripper
        
        :param speed: gripper speed between 0 and 255
        :param force: gripper force between 0 and 255
        :param wait: whether to wait for the robotiq motion to complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 5, only available if wait=True
        
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation 
        """
        return self._arm.robotiq_open(speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)

    def robotiq_close(self, speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs):
        """
        Close the robotiq gripper
        
        :param speed: gripper speed between 0 and 255
        :param force: gripper force between 0 and 255
        :param wait: whether to wait for the robotiq motion to complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 3, only available if wait=True
        
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation
        """
        return self._arm.robotiq_close(speed=speed, force=force, wait=wait, timeout=timeout, **kwargs)

    def robotiq_get_status(self, number_of_registers=3):
        """
        Reading the status of robotiq gripper
        
        :param number_of_registers: number of registers, 1/2/3, default is 3
            number_of_registers=1: reading the content of register 0x07D0
            number_of_registers=2: reading the content of register 0x07D0/0x07D1
            number_of_registers=3: reading the content of register 0x07D0/0x07D1/0x07D2
            
            Note: 
                register 0x07D0: Register GRIPPER STATUS
                register 0x07D1: Register FAULT STATUS and register POSITION REQUEST ECHO
                register 0x07D2: Register POSITION and register CURRENT
        :return: tuple((code, robotiq_response))
            code: See the API code documentation for details.
            robotiq_response: See the robotiq documentation
        """
        return self._arm.robotiq_get_status(number_of_registers=number_of_registers)

    @property
    def robotiq_status(self):
        """
        The last state value obtained
        
        Note:
            1. Successfully call the robotiq related interface with wait parameter (when the parameter wait = True is set) will update this value
            2. Successfully calling interface robotiq_get_status will partially or completely update this value
        
        :return status dict
            {
                'gOBJ': 0,  # Object detection status, is a built-in feature that provides information on possible object pick-up
                'gSTA': 0,  # Gripper status, returns the current status & motion of the Gripper fingers
                'gGTO': 0,  # Action status, echo of the rGTO bit(go to bit)
                'gACT': 0,  # Activation status, echo of the rACT bit(activation bit)
                'kFLT': 0,  # Echo of the requested position for the Gripper
                'gFLT': 0,  # Fault status
                'gPR': 0,  # Echo of the requested position for the Gripper
                'gPO': 0,  # Actual position of the Gripper obtained via the encoders
                'gCU': 0,  # The current is read instantaneously from the motor drive
            }
            Note: -1 means never updated
        """
        return self._arm.robotiq_status

    def set_bio_gripper_enable(self, enable=True, wait=True, timeout=3):
        """
        If not already enabled. Enable the bio gripper
        
        :param enable: enable or not
        :param wait: whether to wait for the bio gripper enable complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 3, only available if wait=True
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_bio_gripper_enable(enable, wait=wait, timeout=timeout)

    def set_bio_gripper_speed(self, speed):
        """
        Set the speed of the bio gripper
        
        :param speed: speed
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_bio_gripper_speed(speed)

    def open_bio_gripper(self, speed=0, wait=True, timeout=5, **kwargs):
        """
        Open the bio gripper
        
        :param speed: speed value, default is 0 (not set the speed)
        :param wait: whether to wait for the bio gripper motion complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 5, only available if wait=True
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.open_bio_gripper(speed=speed, wait=wait, timeout=timeout, **kwargs)

    def close_bio_gripper(self, speed=0, wait=True, timeout=5, **kwargs):
        """
        Close the bio gripper
        
        :param speed: speed value, default is 0 (not set the speed)
        :param wait: whether to wait for the bio gripper motion complete, default is True
        :param timeout: maximum waiting time(unit: second), default is 5, only available if wait=True
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.close_bio_gripper(speed=speed, wait=wait, timeout=timeout, **kwargs)

    def get_bio_gripper_status(self):
        """
        Get the status of the bio gripper
        
        :return: tuple((code, status))
            code: See the API code documentation for details.
            status: status
                status & 0x03 == 0: stop
                status & 0x03 == 1: motion
                status & 0x03 == 2: catch
                status & 0x03 == 3: error
                (status >> 2) & 0x03 == 0: not enabled
                (status >> 2) & 0x03 == 1: enabling
                (status >> 2) & 0x03 == 2: enabled
        """
        return self._arm.get_bio_gripper_status()

    def get_bio_gripper_error(self):
        """
        Get the error code of the bio gripper
        
        :return: tuple((code, error_code))
            code: See the API code documentation for details.
            error_code: See Chapter 7 of the xArm User Manual for details. 
        """
        return self._arm.get_bio_gripper_error()

    def clean_bio_gripper_error(self):
        """
        Clean the error code of the bio gripper
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.clean_bio_gripper_error()

    def set_tgpio_modbus_timeout(self, timeout):
        """
        Set the modbus timeout of the tool gpio
        
        :param timeout: timeout, seconds
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tgpio_modbus_timeout(timeout)

    def set_tgpio_modbus_baudrate(self, baud):
        """
        Set the modbus baudrate of the tool gpio
        
        :param baud: 4800/9600/19200/38400/57600/115200/230400/460800/921600/1000000/1500000/2000000/2500000
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_tgpio_modbus_baudrate(baud)

    def get_tgpio_modbus_baudrate(self):
        """
        Get the modbus baudrate of the tool gpio

        :return: tuple((code, baudrate)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            baudrate: the modbus baudrate of the tool gpio
        """
        return self._arm.get_tgpio_modbus_baudrate()

    def getset_tgpio_modbus_data(self, datas, min_res_len=0):
        """
        Send the modbus data to the tool gpio
        
        :param datas: data_list
        :param min_res_len: the minimum length of modbus response data. Used to check the data length, if not specified, no check
        
        :return: tuple((code, modbus_response))
            code: See the API code documentation for details.
            modbus_response: modbus response data
        """
        return self._arm.getset_tgpio_modbus_data(datas, min_res_len=min_res_len)

    def set_report_tau_or_i(self, tau_or_i=0):
        """
        Set the reported torque or electric current
        
        :param tau_or_i: 
            0: torque
            1: electric current
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_report_tau_or_i(tau_or_i=tau_or_i)

    def get_report_tau_or_i(self):
        """
        Get the reported torque or electric current
        
        :return: tuple((code, tau_or_i))
            code: See the API code documentation for details.
            tau_or_i: 
                0: torque
                1: electric current
        """
        return self._arm.get_report_tau_or_i()

    def set_self_collision_detection(self, on_off):
        """
        Set whether to enable self-collision detection 
        
        :param on_off: enable or not
        
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_self_collision_detection(on_off)

    def set_collision_tool_model(self, tool_type, *args, **kwargs):
        """
        Set the geometric model of the end effector for self collision detection
         
        :param tool_type: the geometric model type
            0: No end effector, no additional parameters required
            1: xArm Gripper, no additional parameters required
            2: xArm Vacuum Gripper, no additional parameters required
            3: xArm Bio Gripper, no additional parameters required
            4: Robotiq-2F-85 Gripper, no additional parameters required
            5: Robotiq-2F-140 Gripper, no additional parameters required
            21: Cylinder, need additional parameters radius, height
                self.set_collision_tool_model(21, radius=45, height=137)
                :param radius: the radius of cylinder, (unit: mm)
                :param height: the height of cylinder, (unit: mm)
            22: Cuboid, need additional parameters x, y, z
                self.set_collision_tool_model(22, x=234, y=323, z=23)
                :param x: the length of the cuboid in the x coordinate direction, (unit: mm)
                :param y: the length of the cuboid in the y coordinate direction, (unit: mm)
                :param z: the length of the cuboid in the z coordinate direction, (unit: mm)
        :param args: additional parameters
        :param kwargs: additional parameters
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_collision_tool_model(tool_type, *args, **kwargs)

    def set_simulation_robot(self, on_off):
        """
        Set the simulation rbt_s
        
        :param on_off: True/False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_simulation_robot(on_off)

    def vc_set_joint_velocity(self, speeds, is_radian=None, is_sync=True, **kwargs):
        """
        Joint velocity control, need to be set to joint velocity control mode(self.set_mode(4))
        Note:
            1. only available if firmware_version >= 1.6.9
        
        :param speeds: [spd_J1, spd_J2, ..., spd_J7]
        :param is_radian: the spd_Jx in radians or not, default is self.default_is_radian
        :param is_sync: whether all joints accelerate and decelerate synchronously, default is True
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.vc_set_joint_velocity(speeds, is_radian=is_radian, is_sync=is_sync, **kwargs)

    def vc_set_cartesian_velocity(self, speeds, is_radian=None, is_tool_coord=False, **kwargs):
        """
        Cartesian velocity control, need to be set to cartesian velocity control mode(self.set_mode(5))
        Note:
            1. only available if firmware_version >= 1.6.9
            
        :param speeds: [spd_x, spd_y, spd_z, spd_rx, spd_ry, spd_rz]
        :param is_radian: the spd_rx/spd_ry/spd_rz in radians or not, default is self.default_is_radian
        :param is_tool_coord: is tool coordinate or not, default is False
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.vc_set_cartesian_velocity(speeds, is_radian=is_radian, is_tool_coord=is_tool_coord, **kwargs)

    def calibrate_tcp_coordinate_offset(self, four_points, is_radian=None):
        """
        Four-point method to calibrate tool coordinate system position offset
        Note:
            1. only available if firmware_version >= 1.6.9

        :param four_points: a list of four teaching coordinate positions [x, y, z, roll, pitch, yaw]
        :param is_radian: the roll/pitch/yaw value of the each point in radians or not, default is self.default_is_radian
        :return: tuple((code, xyz_offset)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            xyz_offset: calculated xyz(mm) TCP offset, [x, y, z] 
        """
        return self._arm.calibrate_tcp_coordinate_offset(four_points, is_radian=is_radian)

    def calibrate_tcp_orientation_offset(self, rpy_be, rpy_bt, input_is_radian=None, return_is_radian=None):
        """
        An additional teaching point to calibrate the tool coordinate system attitude offset
        Note:
            1. only available if firmware_version >= 1.6.9

        :param rpy_be: the rpy value of the teaching point without TCP offset [roll, pitch, yaw]
        :param rpy_bt: the rpy value of the teaching point with TCP offset [roll, pitch, yaw]
        :param input_is_radian: the roll/pitch/yaw value of rpy_be and rpy_bt in radians or not, default is self.default_is_radian
        :param return_is_radian: the roll/pitch/yaw value of result in radians or not, default is self.default_is_radian
        :return: tuple((code, rpy_offset)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            rpy_offset: calculated rpy TCP offset, [roll, pitch, yaw]
        """
        return self._arm.calibrate_tcp_orientation_offset(rpy_be, rpy_bt, input_is_radian=input_is_radian, return_is_radian=return_is_radian)

    def calibrate_user_orientation_offset(self, three_points, mode=0, trust_ind=0, input_is_radian=None, return_is_radian=None):
        """
        Three-point method teaches user coordinate system posture offset
        Note:
            1. only available if firmware_version >= 1.6.9
        
        Note:
            First determine a point in the working space, move along the desired coordinate system x+ to determine the second point,
            and then move along the desired coordinate system y+ to determine the third point. 
            Note that the x+ direction is as accurate as possible. 
            If the y+ direction is not completely perpendicular to x+, it will be corrected in the calculation process.

        :param three_points: a list of teaching TCP coordinate positions [x, y, z, roll, pitch, yaw]
        :param input_is_radian: the roll/pitch/yaw value of the each point in radians or not, default is self.default_is_radian
        :param return_is_radian: the roll/pitch/yaw value of result in radians or not, default is self.default_is_radian
        :return: tuple((code, rpy_offset)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            rpy_offset: calculated rpy user offset, [roll, pitch, yaw]
        """
        return self._arm.calibrate_user_orientation_offset(three_points, mode=mode, trust_ind=trust_ind, input_is_radian=input_is_radian, return_is_radian=return_is_radian)
    
    def calibrate_user_coordinate_offset(self, rpy_ub, pos_b_uorg, is_radian=None):
        """
        An additional teaching point determines the position offset of the user coordinate system.
        Note:
            1. only available if firmware_version >= 1.6.9

        :param rpy_ub: the confirmed offset of the base coordinate system in the user coordinate system [roll, pitch, yaw], which is the result of calibrate_user_orientation_offset()
        :param pos_b_uorg: the position of the teaching point in the base coordinate system [x, y, z], if the arm cannot reach the target position, the user can manually input the position of the target in the base coordinate.
        :param is_radian: the roll/pitch/yaw value of rpy_ub in radians or not, default is self.default_is_radian
        :return: tuple((code, xyz_offset)), only when code is 0, the returned result is correct.
            code: See the API code documentation for details.
            xyz_offset: calculated xyz(mm) user offset, [x, y, z] 
        """
        return self._arm.calibrate_user_coordinate_offset(rpy_ub, pos_b_uorg, is_radian=is_radian)

    def set_impedance(self, coord, c_axis, M, K, B):
        """
        set all parameters of impedance control.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param coord: task frame. 0: base frame. 1: tool frame.
        :param c_axis: a 6d vector of 0s and 1s. 1 means that rbt_s will be impedance in the corresponding axis of the task frame.
        :param M: mass. (kg)
        :param K: stiffness coefficient.
        :param B: damping coefficient. invalid.   Note: the value is set to 2*sqrt(M*K) in controller.
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_impedance(coord, c_axis, M, K, B)

    def set_impedance_mbk(self, M, K, B):
        """
        set mbk parameters of impedance control.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param M: mass. (kg)
        :param K: stiffness coefficient.
        :param B: damping coefficient. invalid.   Note: the value is set to 2*sqrt(M*K) in controller.
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_impedance_mbk(M, K, B)

    def set_impedance_config(self, coord, c_axis):
        """
        set impedance control parameters of impedance control.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param coord: task frame. 0: base frame. 1: tool frame.
        :param c_axis: a 6d vector of 0s and 1s. 1 means that rbt_s will be impedance in the corresponding axis of the task frame.
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_impedance_config(coord, c_axis)

    def config_force_control(self, coord, c_axis, f_ref, limits):
        """
        set force control parameters.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param coord:  task frame. 0: base frame. 1: tool frame.
        :param c_axis: a 6d vector of 0s and 1s. 1 means that rbt_s will be compliant in the corresponding axis of the task frame.
        :param f_ref:  the forces/torques the rbt_s will apply to its environment. The rbt_s adjusts its position along/about compliant axis in
                       order to achieve the specified force/torque.
        :param limits:  for compliant axes, these values are the maximum allowed tcp speed along/about the axis.
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.config_force_control(coord, c_axis, f_ref, limits)

    def set_force_control_pid(self, kp, ki, kd, xe_limit):
        """
        set force control pid parameters.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param kp: proportional gain. default : 0.005
        :param ki: integral gain. default : 0.00006
        :param kd: differential gain. default : 0.0
        :param xe_limit: 6d vector. for compliant axes, these values are the maximum allowed tcp speed along/about the axis. mm/s
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.set_force_control_pid(kp, ki, kd, xe_limit)

    def ft_sensor_set_zero(self):
        """
        set force/torque offset.
        Note:
            1. only available if firmware_version >= 1.7.0

        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.ft_sensor_set_zero()

    def ft_sensor_iden_load(self):
        """
        start load identification.
        Note:
            1. only available if firmware_version >= 1.7.0

        :return: tuple((code, load)) only when code is 0, the returned result is correct.
            code:  See the API code documentation for details.
            load:  [mass,x_centroid,y_centroid,z_centroid,Fx_offset,Fy_offset,Fz_offset,Mx_offset,My_offset,Mz_ffset]
        """
        return self._arm.ft_sensor_iden_load()

    def ft_sensor_cali_load(self, iden_result_list):
        """
        write load parameter value
        Note:
            1. only available if firmware_version >= 1.7.0

        :param iden_result_list:  [mass,x_centroid,y_centroid,z_centroid,Fx_offset,Fy_offset,Fz_offset,Mx_offset,My_offset,Mz_ffset]
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.ft_sensor_cali_load(iden_result_list)

    def ft_sensor_enable(self, on_off):
        """
        used for enabling and disabling the use of external F/T measurements in the controller.
        Note:
            1. only available if firmware_version >= 1.7.0

        :param on_off: enable or disable F/T data sampling.
        :return: code
            code: See the API code documentation for details.
        """
        return self._arm.ft_sensor_enable(on_off)

    def ft_sensor_app_set(self, app_code):
        """
        set rbt_s to be controlled in force mode
        Note:
            1. only available if firmware_version >= 1.7.0

        :param app_code: force mode. 0: non-force mode  1: impendance control  2:force control
        :return: tuple((code, status))
            code: See the API code documentation for details.
            status: 
        """
        return self._arm.ft_sensor_app_set(app_code)

    def ft_sensor_app_get(self):
        """
        get force mode
        Note:
            1. only available if firmware_version >= 1.7.0

        :return: tuple((code, status))
            code: See the API code documentation for details.
            status: 0: non-force mode
                    1: impedance control mode
                    2: force control mode
        """
        return self._arm.ft_sensor_app_get()

    def get_exe_ft(self):
        """
        get extenal force/torque
        Note:
            1. only available if firmware_version >= 1.7.0

        :return: tuple((code, exe_ft))
            code: See the API code documentation for details.
            exe_ft: only when code is 0, the returned result is correct.
        """
        return self._arm.get_exe_ft()
