# Lab 2 Documentation

## Code Structure
The main PID controller code is implemented in the `_compute_desired_force_and_euler()`. We create diagonal matrices for $K_{pos}$, $K_{vel}$, and $K_{pos, i}$ which correspond to the P, D, and I terms respectively.

```
self.K_pos_p = np.diag([20.0, 25.0, 15.0])
self.K_vel_d = np.diag([10.0, 15.0, 10.0])
self.K_pos_i = np.diag([0.4, 0.4, 0.15])
```
We first calculate the desired acceleration (Lines 154-164), then calculate the command thrust (Line 166), and the desired orientation (Lines 168-185) while also setting the desired orientation to unity if the norm of the desired acceleration is 0. We then return these results from the function.

## To Run
Paste the lab2 folder within the aer-course-project folder

```bash
cd /safe-control-gym/aer-course-project/lab2
python3 main.py --overrides ./lab2.yaml
```

