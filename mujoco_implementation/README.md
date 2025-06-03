## Objective
This repository is a collection of several experiments carried out on the Humanoid-v4. The objective of these experiements is to stage the development of control algorithms and workflows that implement walking motions on a bipedal robot. The list of experiments being carried out are:
1. Balancing on one leg using LQR control.
2. Visualizing contact forces for one-leg balancing
3. Impulse response and balance retention
4. Walking - optimizing transitions between 3 stances. 

### Balancing on one leg using LQR Control
This experiment is a [tutorial](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/LQR.ipynb?authuser=2#scrollTo=zJmbOJMurRna) provided in the MuJoCo repository maintained by Google Deepmind. A few changes have been made to better explain the flow of code. LQR control is very straightforward and optimal for linear systems. That's essentially the motivation: there are no linear systems in our world outside the classroom. Therefore, the flow of code is staged as:
1. Linearise the system to inject a controller
    1. Find the equilibrium $x_0$. 
    2. Find the optimal $u_0$ (control force) - aka finding the control setpoint using inverse dynamics
    3. Find the actuator values that give you this $u_0$. Ensure that the actuator values are actually produceable to maintain constraints like 0 acceleration.

2. Develop the Q and R matrices by defining the costs that matter.
   Let the full LQR cost matrix $\( Q \in \mathbb{R}^{2nv \times 2nv} \)$ be defined as:
  ```math
Q = \begin{bmatrix}
Q_{\text{pos}} & 0 \\
0 & Q_{\text{vel}}
\end{bmatrix}
= 
\begin{bmatrix}
Q_{\text{balance}} + Q_{\text{joints}} & 0 \\
0 & 0
\end{bmatrix}
  ```
and 
```math
R = I_{2nv x 2nv}
```
4. Obtain the K (gain) matrix that is to close the system loop.
```math
   K = (R + B^T  P B)^{-1} B^T P A
```
   Note that P is the cost matrix that we are solving for using Algebraic Riccati Equations. This is slightly modified into the form where $M = PA$ and we solve the expression:
   ```math
   \dot M = 0 = MA + A^TM- MBR^{-1}B^T M^T + Q 
   ``` 
#### Results 

