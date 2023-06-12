# autonomous-duckie

Scripts to generate control files for map1.

# Guide
Add `.py` files into the `CS4278-5478-Project-Materials/` folder to run `train2.py`.

# Hyperparameter tuning
Keep the rest of the hyperparameters the same except the following.

## Training for the first time
```
restore = False
max_episodes = 15000
min_buffer = 99000
buffer_limit = 990000
epsilon_decay = 32000
bias = 10 # in compute_epsilon()
```

## Second train routine
```
# restore run_15000.pt from earlier
restore = True
max_episodes = TBC
min_buffer = 99000
buffer_limit = 100000
epsilon_decay = 10000
bias = 15000 # in compute_epsilon()
```

# To generate control files
Run `test.py`.

# To render control files.
Run `test_render.py`.
