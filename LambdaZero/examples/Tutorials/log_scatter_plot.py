import numpy as np
import wandb

wandb.init(project='log_scatter_test')

x_values = np.random.randn(1, 100).ravel().tolist()
y_values = np.random.randn(1, 100).ravel().tolist()

data = [[x, y] for (x, y) in zip(x_values, y_values)]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_custom_plot_id" : wandb.plot.scatter(table, "x", "y", title="Custom Y vs X Scatter Plot")})