# 1. initialize proxy




# class BoltzmannSearchAgent
#   init
#       self.env
#       self.reward = proxyReward
#
#   def boltzmann_step():
#       states = [env.state_for a in action_mask]
#       rewards = [self.reward(state) for state in states]
#       action = sample(softmax(rewards), temperature)
#       self.env.step(action)
#       if max_steps: env.reset()
#       return None


# BoltzmannSearchTrainer() ~~ PPOTrainer