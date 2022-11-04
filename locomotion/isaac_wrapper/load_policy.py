import torch
from locomotion.isaac_wrapper.skill_actor_critic import SkillActorCritic
from locomotion.isaac_wrapper.residual_actor_critic import ResidualActorCritic
from locomotion.isaac_wrapper.multiskill_actor_critic import MultiSkillActorCritic
def get_walking_policy():
    # run_name = "a1_flat/Jun21_17-35-58_" #walking
    run_name = "a1_flat/Sep26_17-16-44_" # High P walking
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_1550.pt"
    loaded_dict = torch.load(path)
    actor_critic = SkillActorCritic(45, 45, 12, [512, 256, 128], [512, 256, 128]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

def get_crouching_policy():
    # run_name = "crouching/Jun22_15-47-56_/model_1500.pt" # crouching
    run_name = "crouching/Aug21_22-24-48_/model_6000.pt"
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
            "scaled_base_ang_vel": 3,
            "projected_gravity": 3,
            "object_position": 1,
            "relative_dof": 12,
            "scaled_dof_vel": 12,
            "actions": 12}
    actor_obs =[["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "relative_dof",
                "scaled_dof_vel",
                "actions"],
                
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                ]
    critic_obs = [                    
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                    ]
    actor_critic = ResidualActorCritic(num_skills=2,
                                       obs_sizes=obs_sizes,
                                       actor_obs=actor_obs,
                                       critic_obs=critic_obs,
                                       num_actions=12,
                                       actor_hidden_dims=[[512, 256, 128], [256, 128]], 
                                       critic_hidden_dims=[512, 256, 128],
                                       weight_network_dims=[512,256]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

def get_turning_policy():
    # /home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs//model_100.pt
    # run_name = "turning/Jun24_13-41-47_" # Turn right
    # run_name = "turning/Sep20_18-27-10_" # Turn right
    # run_name = "turning/Sep21_19-19-47_"
    # run_name = "turning/Sep21_19-53-24_"
    # run_name = "turning/Sep22_14-33-00_"
    # run_name = "turning/Sep26_15-28-10_"
    # run_name = "turning/Sep26_15-52-16_"
    run_name = "turning/Sep26_16-17-42_"
    
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_150.pt"
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
            "scaled_base_ang_vel": 3,
            "projected_gravity": 3,
            # "object_position": 2,
            "relative_dof": 12,
            "scaled_dof_vel": 12,
            "actions": 12}
    actor_obs =[["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "relative_dof",
                "scaled_dof_vel",
                "actions"],
                
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                # "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                ]
    critic_obs = [                    
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                # "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                    ]
    actor_critic = ResidualActorCritic(num_skills=2,
                                       obs_sizes=obs_sizes,
                                       actor_obs=actor_obs,
                                       critic_obs=critic_obs,
                                       num_actions=12,
                                       actor_hidden_dims=[[512, 256, 128], [256, 128]], 
                                       critic_hidden_dims=[512, 256, 128],
                                       weight_network_dims=[128]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

def get_dooropen_policy():
    # /home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs//model_100.pt
    # run_name = "turning/Jun24_13-41-47_" # Turn right
    # run_name = "turning/Sep20_18-27-10_" # Turn right
    # run_name = "turning/Sep21_19-19-47_"
    # run_name = "turning/Sep21_19-53-24_"
    # run_name = "turning/Sep22_14-33-00_"
    # run_name = "turning/Sep26_15-28-10_"
    # run_name = "turning/Sep26_15-52-16_"
    # run_name = "dooropen/Jul11_13-04-00_/model_1500.pt"
    run_name = "dooropen/Oct14_14-40-56_/model_3000.pt"
    
    
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
            "scaled_base_ang_vel": 3,
            "projected_gravity": 3,
            "door_state": 2,
            "door_angle": 1,
            # "object_position": 2,
            "relative_dof": 12,
            "scaled_dof_vel": 12,
            "actions": 12}
    actor_obs =[["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "relative_dof",
                "scaled_dof_vel",
                "actions"],
                
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "door_state",
                "door_angle",
                # "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                ]
    critic_obs = [                    
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "door_state",
                "door_angle",
                # "object_position",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                    ]
    actor_critic = ResidualActorCritic(num_skills=2,
                                       obs_sizes=obs_sizes,
                                       actor_obs=actor_obs,
                                       critic_obs=critic_obs,
                                       num_actions=12,
                                       actor_hidden_dims=[[512, 256, 128], [512, 256]], 
                                       critic_hidden_dims=[512, 256, 128],
                                       weight_network_dims=[512, 256, 128]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

def get_target_reach_policy():
    # run_name = "multiskill_targetreach/Sep28_20-15-41_"
    run_name = "multiskill_targetreach/Sep30_10-28-01_"
    # run_name = "multiskill_targetreach/Jul07_10-24-55_"
    
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_5500.pt"
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
                    "scaled_base_ang_vel": 3,
                    "projected_gravity": 3,
                    "target_position": 2,
                    "relative_dof": 12,
                    "scaled_dof_vel": 12,
                    "actions": 12}
    actor_obs =[["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],
                    
                    ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"],

                    # ["scaled_base_lin_vel",
                    # "scaled_base_ang_vel",
                    # "projected_gravity",
                    # "target_position",
                    # "relative_dof",
                    # "scaled_dof_vel",
                    # "actions"]
                ]
    critic_obs = [                    
                ["scaled_base_lin_vel",
                    "scaled_base_ang_vel",
                    "projected_gravity",
                    "target_position",
                    "relative_dof",
                    "scaled_dof_vel",
                    "actions"]
                    ]
    actor_critic = MultiSkillActorCritic(num_skills=4,
                                       obs_sizes=obs_sizes,
                                       actor_obs=actor_obs,
                                       critic_obs=critic_obs,
                                       num_actions=12,
                                       actor_hidden_dims=[[512, 256, 128], [256, 128], [256, 128], [256, 128]], 
                                       critic_hidden_dims=[512, 256, 128],
                                       weight_network_dims=[512, 256, 128]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic
    
def get_pushing_policy():
    run_name = "multiskill_object_push/Sep20_15-29-07_/model_2500.pt"
    # run_name = "multiskill_object_push/Nov02_16-53-08_/model_1500.pt"
    run_name = "multiskill_object_push/Nov02_22-20-21_/model_9000.pt"
    
    
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
            "scaled_base_ang_vel": 3,
            "projected_gravity": 3,
            "object_location": 2,
            "target_location":2,
            # "object_position": 2,
            "relative_dof": 12,
            "scaled_dof_vel": 12,
            "actions": 12}
    actor_obs =[["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "relative_dof",
                "scaled_dof_vel",
                "actions"],
                
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "object_location",
                "target_location",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                ]
    critic_obs = [                    
                ["scaled_base_lin_vel",
                "scaled_base_ang_vel",
                "projected_gravity",
                "object_location",
                "target_location",
                "relative_dof",
                "scaled_dof_vel",
                "actions"]
                    ]
    actor_critic = ResidualActorCritic(num_skills=2,
                                       obs_sizes=obs_sizes,
                                       actor_obs=actor_obs,
                                       critic_obs=critic_obs,
                                       num_actions=12,
                                       actor_hidden_dims=[[512, 256, 128], [512,256, 128]], 
                                       critic_hidden_dims=[512, 256, 128],
                                       weight_network_dims=[512,256]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

if __name__ == '__main__':
    policy = get_policy()