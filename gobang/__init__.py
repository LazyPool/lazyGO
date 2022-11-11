from gym.envs.registration import register

register(
    id='lazyGO-v0',
    entry_point='gobang.envs:lazyGO',
    max_episode_steps=300,
)
