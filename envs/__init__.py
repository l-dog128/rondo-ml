from gymnasium.envs.registration import register
register(
  id="Rondo-V0",
  entry_point="envs.rondo_env:RondoEnv"
)