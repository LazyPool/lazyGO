from gym.envs.registration import register

register(
    id='lazyGO-v0',
    entry_point='lazyGO-v0',
    max_episode_steps=300,
)
