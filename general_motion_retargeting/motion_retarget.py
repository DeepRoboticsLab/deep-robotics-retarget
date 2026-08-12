"""Core IK-based motion retargeting module.

This module implements the GeneralMotionRetargeting class, which uses inverse kinematics
(via the Mink library) to retarget human body tracking data to humanoid robot joint
configurations. It supports two-stage IK solving with collision avoidance and joint limits.
"""

import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting using two-stage inverse kinematics.

    This class retargets human motion data to a humanoid robot by solving a two-stage
    IK problem:
        - Stage 1 (Table 1): Orientation-only tracking for end-effectors.
        - Stage 2 (Table 2): Full position + orientation tracking for all body parts.

    The human body is scaled to match the robot's proportions using a configurable
    scale table. Collision avoidance and joint limits can be optionally enabled.

    Supported human data sources: SMPLX, BVH (Lafan1, Nokov), FBX (OptiTrack).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp",
        damping: float=5e-1,
        verbose: bool=True,
        use_velocity_limit: bool=False,
        use_collision_avoidance: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.hand_tip_offset = ik_config.get("hand_tip_offset", 0.0)  # wrist joint to fake hand tip distance (m)

        self.max_iter_t1 = 10   # Table1: orientation-only
        self.max_iter_t2 = 100  # Table2: position tracking, more iterations needed
        self.convergence_threshold_t2 = 1e-5  # Tighter convergence for table2

        self.solver = solver
        self.damping = damping
        
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            # Use joint names (not motor/actuator names) for mink.VelocityLimit
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_dof_names.keys() if k is not None}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        if use_collision_avoidance:
            collision_pairs = self._build_collision_pairs()
            self.ik_limits.append(
                mink.CollisionAvoidanceLimit(
                    model=self.model,
                    geom_pairs=collision_pairs,
                    gain=0.85,
                    minimum_distance_from_collisions=0.05,
                    collision_detection_distance=0.1,
                )
            )
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        # Offset hand target z by hand_tip_offset so fake hand tip matches BVH hand position.
        # Then clamp to hand_tip_offset so the hand tip never goes below floor.
        hand_clamp_names = ["LeftHand", "RightHand"]
        min_hand_z = self.hand_tip_offset + 0.01  # wrist Z floor = hand length, so hand tip >= 0
        for hn in hand_clamp_names:
            if hn in human_data:
                pos = human_data[hn][0]
                pos[2] = pos[2] + self.hand_tip_offset  # shift wrist target up by hand length
                if pos[2] < min_hand_z:
                    pos[2] = min_hand_z

        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            
            
    def retarget(self, human_data, offset_to_ground=False):
        """Solve IK to retarget human motion to robot.
        
        Args:
            human_data: Dict mapping body names to (position, quaternion) tuples.
            offset_to_ground: Whether to offset human data so feet touch ground.
            
        Returns:
            Robot joint positions (qpos) as numpy array.
        """
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, limits=self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter_t1:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, limits=self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, limits=self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            best_error = min(curr_error, next_error)
            no_improve_count = 0
            while num_iter < self.max_iter_t2:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, limits=self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                num_iter += 1
                if next_error < best_error - self.convergence_threshold_t2:
                    best_error = next_error
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= 5 and next_error > 0.01:
                    break
                if next_error < self.convergence_threshold_t2:
                    break

        return self.configuration.data.qpos.copy()


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            rot_offset = rot_offsets.get(body_name, R.identity())
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offset).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets.get(body_name, np.zeros(3))
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data

    def _build_collision_pairs(self):
        """Build collision pairs for the robot model"""
        collision_pairs = []

        # Build a mapping: body_name -> list of geom IDs
        body_name_to_geom_ids = {}
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id)
            if body_name is not None:
                body_name_to_geom_ids.setdefault(body_name, []).append(geom_id)

        def get_geom_ids(body_names):
            ids = []
            for name in body_names:
                ids.extend(body_name_to_geom_ids.get(name, []))
            return ids

        # 1. Self-collision: left arm vs right arm
        left_arm_bodies = [
            "left_shoulder_y_link", "left_shoulder_x_link",
            "left_shoulder_z_link", "left_elbow_link",
            "left_wrist_z_link", "left_wrist_y_link", "left_wrist_x_link",
        ]
        right_arm_bodies = [
            "right_shoulder_y_link", "right_shoulder_x_link",
            "right_shoulder_z_link", "right_elbow_link",
            "right_wrist_z_link", "right_wrist_y_link", "right_wrist_x_link",
        ]
        left_arm_gids = get_geom_ids(left_arm_bodies)
        right_arm_gids = get_geom_ids(right_arm_bodies)
        if left_arm_gids and right_arm_gids:
            collision_pairs.append((left_arm_gids, right_arm_gids))

        # 2. Arms vs body/torso
        body_bodies = ["body", "waist_z_link", "waist_x_link"]
        body_gids = get_geom_ids(body_bodies)
        if body_gids:
            if left_arm_gids:
                collision_pairs.append((left_arm_gids, body_gids))
            if right_arm_gids:
                collision_pairs.append((right_arm_gids, body_gids))

        # 3. Arms vs ground (floor)
        all_arm_gids = left_arm_gids + right_arm_gids
        floor_gids = get_geom_ids(["world"])
        # Also try to find the floor geom by name
        try:
            floor_geom_id = self.model.geom("floor").id
            floor_gids = list(set(floor_gids + [floor_geom_id]))
        except Exception:
            pass
        if floor_gids:
            if all_arm_gids:
                collision_pairs.append((all_arm_gids, floor_gids))

        # 4. Legs geom IDs
        left_leg_bodies = [
            "left_hip_y_link", "left_hip_x_link", "left_hip_z_link",
            "left_knee_link", "left_ankle_y_link", "left_ankle_x_link",
        ]
        right_leg_bodies = [
            "right_hip_y_link", "right_hip_x_link", "right_hip_z_link",
            "right_knee_link", "right_ankle_y_link", "right_ankle_x_link",
        ]
        left_leg_gids = get_geom_ids(left_leg_bodies)
        right_leg_gids = get_geom_ids(right_leg_bodies)

        # 5. Arms vs legs
        all_leg_gids = left_leg_gids + right_leg_gids
        if all_arm_gids and all_leg_gids:
            collision_pairs.append((all_arm_gids, all_leg_gids))

        # 6. Arms vs head/neck
        head_bodies = ["head_link", "neck_link"]
        head_gids = get_geom_ids(head_bodies)
        if head_gids:
            if left_arm_gids:
                collision_pairs.append((left_arm_gids, head_gids))
            if right_arm_gids:
                collision_pairs.append((right_arm_gids, head_gids))

        return collision_pairs