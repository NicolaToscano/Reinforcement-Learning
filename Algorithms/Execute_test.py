from Test import TEST
n_episodes = 100
env_name = "Environments/env1/Unity Environment"
model = "Saved Models/Env1_ppo_model.pth"
agent = TEST(n_episodes, env_name, model)