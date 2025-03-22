"""
Configuration file for "fixed arm" (FA) humanoid environment
with potential-based rewards implemented
"""

import torch
from gpugym.envs.base.legged_robot_config \
    import LeggedRobotCfg, LeggedRobotCfgPPO
from gpugym.envs.base.base_config import BaseConfig

class HumanoidBotElfCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 76
        num_actions = 20
        episode_length_s = 5

    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        #terrain_proportions = [0.1, 0.8, 0., 0., 0.1]

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 5.
        heading_command = False
        ang_vel_command = True

        class ranges:
            # TRAINING COMMAND RANGES #
            lin_vel_x = [0, 3.0]        # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]     # min max [rad/s]
            heading = [0., 0.]

            # PLAY COMMAND RANGES #
            # lin_vel_x = [3., 3.]    # min max [m/s]
            # lin_vel_y = [-0., 0.]     # min max [m/s]
            # ang_vel_yaw = [2, 2]      # min max [rad/s]
            # heading = [0, 0]

    class init_state(LeggedRobotCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 0.9143]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [0.94, 0.94],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.4, .4],
            [-.4, .4],
            [-.2, .4],
            [-.2, .2],
            [-.2, .2],
            [-.2, .2]
        ]

        default_joint_angles = {
            'l_hip_z_joint': 0.,
            'l_hip_x_joint': 0.,
            'l_hip_y_joint': -0.2,
            'l_knee_y_joint': 0.25,  # 0.6
            'l_ankle_y_joint': -0.,
            'l_ankle_x_joint': 0.,
            'l_shld_y_joint': 0.5,
            'l_shld_x_joint': 0.1,
            'l_shld_z_joint': 0.,
            'l_elb_y_joint': -1.2,
            
            'r_hip_z_joint': 0.,
            'r_hip_x_joint': 0.,
            'r_hip_y_joint': -0.2,
            'r_knee_y_joint': 0.25,  # 0.6
            'r_ankle_y_joint': -0.,
            'r_ankle_x_joint': 0.,
            'r_shld_y_joint': 0.5,
            'r_shld_x_joint': -0.1,
            'r_shld_z_joint': 0.,
            'r_elb_y_joint': -1.2,
        }
        
        dof_pos_range = {
            'l_hip_z_joint':  [-0.1, 0.1],
            'l_hip_x_joint':  [-0.06, 0.06],
            'l_hip_y_joint':  [-0.2, 0.2],
            'l_knee_y_joint': [0.1, 0.5],
            'l_ankle_y_joint':[-0.3, 0.3],
            'l_ankle_x_joint':[-0.1, 0.1],
            'l_shld_y_joint': [0.3, 0.7],
            'l_shld_x_joint': [0.1, 0.5],
            'l_shld_z_joint': [-0.1, 0.1],
            'l_elb_y_joint':  [-1.2, -0.8],
            
            'r_hip_z_joint':  [-0.1, 0.1],
            'r_hip_x_joint':  [-0.06, 0.06],
            'r_hip_y_joint':  [-0.2, 0.2],
            'r_knee_y_joint': [0.1, 0.5],
            'r_ankle_y_joint':[-0.3, 0.3],
            'r_ankle_x_joint':[-0.1, 0.1],
            'r_shld_y_joint': [0.3, 0.7],
            'r_shld_x_joint': [-0.5, -0.1],
            'r_shld_z_joint': [-0.1, 0.1],
            'r_elb_y_joint':  [-1.2, -0.8],
        }

        dof_vel_range = {
            'l_hip_z_joint': [-0.1, 0.1],
            'l_hip_x_joint': [-0.1, 0.1],
            'l_hip_y_joint': [-0.1, 0.1],
            'l_knee_y_joint': [-0.1, 0.1],
            'l_ankle_y_joint': [-0.1, 0.1],
            'l_ankle_x_joint': [-0.1, 0.1],
            'l_shld_y_joint': [-0.1, 0.1],
            'l_shld_x_joint': [-0.1, 0.1],
            'l_shld_z_joint': [-0.1, 0.1],
            'l_elb_y_joint':  [-0.1, 0.1],
            
            'r_hip_z_joint': [-0.1, 0.1],
            'r_hip_x_joint': [-0.1, 0.1],
            'r_hip_y_joint': [-0.1, 0.1],
            'r_knee_y_joint': [-0.1, 0.1],
            'r_ankle_y_joint': [-0.1, 0.1],
            'r_ankle_x_joint': [-0.1, 0.1],
            'r_shld_y_joint': [-0.1, 0.1],
            'r_shld_x_joint': [-0.1, 0.1],
            'r_shld_z_joint': [-0.1, 0.1],
            'r_elb_y_joint':  [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            'l_hip_z_joint': 10.,
            'l_hip_x_joint': 10.,
            'l_hip_y_joint': 30.,
            'l_knee_y_joint': 30.,
            'l_ankle_y_joint': 5.,
            'l_ankle_x_joint': 5.,
            'l_shld_y_joint': 10.,
            'l_shld_x_joint': 10.,
            'l_shld_z_joint': 10.,
            'l_elb_y_joint': 10.,
            
            
            'r_hip_z_joint': 10.,
            'r_hip_x_joint': 10.,
            'r_hip_y_joint': 30.,
            'r_knee_y_joint': 30.,
            'r_ankle_y_joint': 5.,
            'r_ankle_x_joint': 5.,
            'r_shld_y_joint': 10.,
            'r_shld_x_joint': 10.,
            'r_shld_z_joint': 10.,
            'r_elb_y_joint': 10.,
        }
        
        damping = {
            'l_hip_z_joint': 1.5,
            'l_hip_x_joint': 1.5,
            'l_hip_y_joint': 2.5,
            'l_knee_y_joint': 2.5,
            'l_ankle_y_joint': 1.,
            'l_ankle_x_joint': 1.,
            'l_shld_y_joint': 1.5,
            'l_shld_x_joint': 1.5,
            'l_shld_z_joint': 1.5,
            'l_elb_y_joint': 1.5,
            
            'r_hip_z_joint': 1.5,
            'r_hip_x_joint': 1.5,
            'r_hip_y_joint': 2.5,
            'r_knee_y_joint': 2.5,
            'r_ankle_y_joint': 1.,
            'r_ankle_x_joint': 1.,
            'r_shld_y_joint': 1.5,
            'r_shld_x_joint': 1.5,
            'r_shld_z_joint': 1.5,
            'r_elb_y_joint': 1.5,
        }

        action_scale = 1.0
        exp_avg_decay = None
        decimation = 10
         
        control_type = 'P' # P: position, V: velocity, T: torques, PT

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-1., 1.]

        push_robots = False
        push_type = "force" # "force", "vel", "mass?" 
        max_push_force = 5. # [N]
        push_interval_s = 3.
        max_push_vel_xy = 1.

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/'\
            'resources/robots/bot_elf/urdf/bot_elf_ess_collision.urdf'
        keypoints = ["base_link"]
        end_effectors = ['l_ankle_x_link', 'r_ankle_x_link']
        foot_name = 'ankle_x_link'
        terminate_after_contacts_on = [
            'base_link',
        ]
        penalize_contacts_on = ['hip_y_link', 'knee_y_link']
        disable_gravity = False
        disable_actions = False
        disable_motors = False

        # (1: disable, 0: enable...bitwise filter)
        self_collisions = 0
        collapse_fixed_joints = True
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True

        # Check GymDofDriveModeFlags
        # (0: none, 1: pos tgt, 2: vel target, 3: effort)
        default_dof_drive_mode = 3

    class rewards(LeggedRobotCfg.rewards):
        pass
        #! "Incorrect" specification of height
        base_height_target = 0.84
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8

        # negative total rewards clipped at zero (avoids early termination)
        only_positive_rewards = False
        tracking_sigma = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            pass
        #     # * "True" rewards * #
        #     action_rate = -1.e-5
        #     action_rate2 = -1.e-6
        #     tracking_lin_vel = 10.
        #     tracking_ang_vel = 5.
        #     torques = -1e-6
        #     dof_pos_limits = -5.0
        #     torque_limits = -1e-3
        #     termination = -100
        #     collision = -1.
        #     no_fly = 0.

        #     # * Shaping rewards * #
        #     # Sweep values: [0.5, 2.5, 10, 25., 50.]
        #     # Default: 5.0
        #     # orientation = 5.0

        #     # Sweep values: [0.2, 1.0, 4.0, 10., 20.]
        #     # Default: 2.0
        #     # base_height = 2.0

        #     # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
        #     # Default: 1.0
        #     # joint_regularization = 1.0

        #     # * PBRS rewards * #
        #     # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
        #     # Default: 1.0
        #     ori_pb = 1.0

        #     # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
        #     # Default: 1.0
        #     baseHeight_pb = 1.0

        #     # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
        #     # Default: 1.0
        #     jointReg_pb = 1.0

    class normalization(LeggedRobotCfg.normalization):           
        class obs_scales:
            base_z = 1.0
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 10.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            base_z = 0.05
            dof_pos = 0.005
            dof_vel = 0.01
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 2
        gravity = [0., 0., -9.81]

        class physx:
            max_depenetration_velocity = 10.0

class HumanoidBotElfFixArmCfg(HumanoidBotElfCfg):
    class env(HumanoidBotElfCfg.env):
        num_envs = 4096
        num_observations = 52
        num_actions = 12
        episode_length_s = 10
    
    class init_state(HumanoidBotElfCfg.init_state):
        reset_mode = 'reset_to_range'
        penetration_check = False
        pos = [0., 0., 0.93]        # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]

        # ranges for [x, y, z, roll, pitch, yaw]
        root_pos_range = [
            [0., 0.],
            [0., 0.],
            [0.93, 0.95],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10],
            [-torch.pi/10, torch.pi/10]
        ]

        # ranges for [v_x, v_y, v_z, w_x, w_y, w_z]
        root_vel_range = [
            [-.4, .4],
            [-.4, .4],
            [-.2, .4],
            [-.2, .2],
            [-.2, .2],
            [-.2, .2]
        ]

        default_joint_angles = {
            'l_hip_z_joint': 0.,
            'l_hip_x_joint': 0.,
            'l_hip_y_joint': -0.2,
            'l_knee_y_joint': 0.25,  # 0.6
            'l_ankle_y_joint': -0.,
            'l_ankle_x_joint': 0.,
            
            'r_hip_z_joint': 0.,
            'r_hip_x_joint': 0.,
            'r_hip_y_joint': -0.2,
            'r_knee_y_joint': 0.25,  # 0.6
            'r_ankle_y_joint': -0.,
            'r_ankle_x_joint': 0.,
        }

        dof_pos_range = {
            'l_hip_z_joint':  [-0.1, 0.1],
            'l_hip_x_joint':  [-0.2, 0.2],
            'l_hip_y_joint':  [-0.2, 0.2],
            'l_knee_y_joint': [0.3, 0.6],
            'l_ankle_y_joint':[-0.3, 0.3],
            'l_ankle_x_joint':[-0.1, 0.1],
            
            'r_hip_z_joint':  [-0.1, 0.1],
            'r_hip_x_joint':  [-0.2, 0.2],
            'r_hip_y_joint':  [-0.2, 0.2],
            'r_knee_y_joint': [0.3, 0.6],
            'r_ankle_y_joint':[-0.3, 0.3],
            'r_ankle_x_joint':[-0.1, 0.1],
        }

        dof_vel_range = {
            'l_hip_z_joint': [-0.1, 0.1],
            'l_hip_x_joint': [-0.1, 0.1],
            'l_hip_y_joint': [-0.1, 0.1],
            'l_knee_y_joint': [-0.1, 0.1],
            'l_ankle_y_joint': [-0.1, 0.1],
            'l_ankle_x_joint': [-0.1, 0.1],
            
            'r_hip_z_joint': [-0.1, 0.1],
            'r_hip_x_joint': [-0.1, 0.1],
            'r_hip_y_joint': [-0.1, 0.1],
            'r_knee_y_joint': [-0.1, 0.1],
            'r_ankle_y_joint': [-0.1, 0.1],
            'r_ankle_x_joint': [-0.1, 0.1],
        }

    class control(LeggedRobotCfg.control):
        # stiffness and damping for joints
        stiffness = {
            'l_hip_z_joint': 25.,
            'l_hip_x_joint': 25.,
            'l_hip_y_joint': 30.,
            'l_knee_y_joint': 40.,
            'l_ankle_y_joint': 3.,
            'l_ankle_x_joint': 3.,            
            
            'r_hip_z_joint': 25.,
            'r_hip_x_joint': 25.,
            'r_hip_y_joint': 30.,
            'r_knee_y_joint': 40.,
            'r_ankle_y_joint': 3.,
            'r_ankle_x_joint': 3.,
        }
        damping = {
            'l_hip_z_joint': 2.5,
            'l_hip_x_joint': 2.5,
            'l_hip_y_joint': 3.,
            'l_knee_y_joint': 4.,
            'l_ankle_y_joint': 0.3,
            'l_ankle_x_joint': 0.3,
            
            'r_hip_z_joint': 2.5,
            'r_hip_x_joint': 2.5,
            'r_hip_y_joint': 3.,
            'r_knee_y_joint': 4.,
            'r_ankle_y_joint': 0.3,
            'r_ankle_x_joint': 0.3,
        }

        action_scale = 1.0
        exp_avg_decay = None
        decimation = 10
         
        control_type = 'P' # P: position, V: velocity, T: torques, PT

    class asset(HumanoidBotElfCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/'\
            'resources/robots/bot_elf/urdf/bot_elf_ess_collision_fixarm.urdf'

    class rewards:
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.5 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9  # ! may want to turn this off
        base_height_target = 0.82
        max_contact_force = 300. # forces above this value are penalized
        
        class scales:
            # * "True" rewards * #
            # action_rate = -2.e-4
            # action_rate2 = -2.e-5
            tracking_lin_vel = 10.
            tracking_ang_vel = 5.
            # torques = -2e-5
            # dof_pos_limits = -2.0
            # torque_limits = -1e-2
            # termination = -20.
            collision = -1.
            # no_fly = 0.5
            # feet_air_time = 1e-6
            # balance_air_time = -1.0

            # * PBRS rewards * #
            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            ori_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            baseHeight_pb = 1.0

            # Sweep values: [0.1, 0.5, 2.0, 5.0, 10.]
            # Default: 1.0
            jointReg_pb_fixarm = 1.0
    
class HumanoidBotElfCfgPPO(LeggedRobotCfgPPO):
    do_wandb = False
    seed = -1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # algorithm training hyperparameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4    # minibatch size = num_envs*nsteps/nminibatches
        learning_rate = 1.e-5
        schedule = 'adaptive'   # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 64
        max_iterations = 1000
        run_name = 'test_1'
        experiment_name = 'BotElf_test'
        #experiment_name = 'PBRS_HumanoidLocomotion'
        save_interval = 50
        plot_input_gradients = False
        plot_parameter_gradients = False
        
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None

    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 256, 256, 128]
        critic_hidden_dims = [128, 256, 256, 128]
        # (elu, relu, selu, crelu, lrelu, tanh, sigmoid)
        activation = 'lrelu'

    class dynamic_model():
        dym_path = None
        
    class lyapunov_network():
        lya_path = None
     
class HumanoidBotElfFixArmCfgPPO(HumanoidBotElfCfgPPO):
    class runner(HumanoidBotElfCfgPPO.runner):
        experiment_name = 'BotElf_FixArm'
        
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
   
