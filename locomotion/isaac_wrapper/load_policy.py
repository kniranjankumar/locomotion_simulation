import torch
from locomotion.isaac_wrapper.skill_actor_critic import SkillActorCritic
from locomotion.isaac_wrapper.residual_actor_critic import ResidualActorCritic

def get_walking_policy():
    run_name = "a1_flat/Jun21_17-35-58_" #walking
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_1500.pt"
    loaded_dict = torch.load(path)
    actor_critic = SkillActorCritic(45, 45, 12, [512, 256, 128], [512, 256, 128]) 
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.to(torch.device('cuda'))
    return actor_critic.act_inference, actor_critic

def get_crouching_policy():
    run_name = "crouching/Jun22_15-47-56_" # crouching
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_1500.pt"
    loaded_dict = torch.load(path)
    obs_sizes = {"scaled_base_lin_vel": 3,
            "scaled_base_ang_vel": 3,
            "projected_gravity": 3,
            "object_position": 2,
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
    run_name = "turning/Sep22_14-33-00_"
    
    path = "/home/niranjan/Projects/Fetch/curious_dog_isaac/legged_gym/logs/"+run_name+"/model_300.pt"
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

if __name__ == '__main__':
    policy = get_policy()