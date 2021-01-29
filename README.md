# car-racing
DQN solution to OpenAI gym CarRacing-v0

## Branches

- ```main``` is a full double DQN with possible use of the DQNReg loss which doesn't seem to converge depsite all of our attempts
- ```cartpole-v1``` is an adaptation to the cartpole environement which is solved by our algorithm
- ```redo-simple``` is a simple DQN very close to the pytorch DQN tutorial which we tried to make to see if it worked on the Carracing environement, which it doesn't
- ```redo-tf``` is the same simple DQN algorithm but implement with TensorFlow  which also doesn't converge

## Files


### To train

All of the hyperparameters can be found in the file ```src/run.py```, the structure should be self explanatory.
```bash
python src/run.py
```

### To play

Change the path of the model you want to load in ```src/play.py```, then run:
```bash
python src/play.py
```

### Debug

There is a lof of information logged through Tensorboard which can be used to debug or see the results.
Furthermore, in ```src/run.py``` in the fucntion ```optimize_model``` one can add the following lines:
```python
optimizer.step()
debug.plot_gradient_flow(model.named_parameters())
total_loss += loss.item()

```
