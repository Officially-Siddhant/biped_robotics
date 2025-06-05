import numpy as np
import mujoco
import scipy.linalg
from HW3.Mujoco.lqr_mujoco import model

class CostMatrix(object):
    def __init__(self, model = model, data = data, qpos0):
        # Class Parameters
        self.model = model
        self.data = data
        self.nu = model.nu # number of actuators
        self.nv = model.nv # number of DOFs
        self.qpos0 = qpos0

        # Cost Coefficients
        self.BALANCE_COST = 1000
        self.BALANCE_JOINT_COST = 3
        self.OTHER_JOINT_COST = 0.3

        # Other params
        self.epsilon = 1e-6

    def set_R(self):
        R = np.eye(3) # every state is activated/penalized.
        return R

    def set_Q(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos = self.qpos0
        mujoco.mj_forward(self.model,self.data)
        jac_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(model,self.data, jac_com, self.model.body('torso').id)

        # Get the Jacobian for the left foot.
        jac_foot = np.zeros((3, self.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jac_foot, None, self.model.body('foot_left').id)
        jac_diff = jac_com - jac_foot

        # Construct the Qbalance matrix
        Qbalance = jac_diff.T @ jac_diff

        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        root_dofs = range(6)
        body_dofs = range(6, nv)
        abdomen_dofs = [
            self.model.joint(name).dofadr[0]
            for name in joint_names
            if 'abdomen' in name
               and not 'z' in name
        ]

        left_leg_dofs = [
            self.model.joint(name).dofadr[0]
            for name in joint_names
            if 'left' in name
               and ('hip' in name or 'knee' in name or 'ankle' in name)
               and not 'z' in name
        ]
        balance_dofs = abdomen_dofs + left_leg_dofs
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)

        # Construct the Qjoint matrix.
        Qjoint = np.eye(self.nv)
        Qjoint[root_dofs, root_dofs] *= 1  # Don't penalize free joint directly.
        Qjoint[balance_dofs, balance_dofs] *= self.BALANCE_JOINT_COST
        Qjoint[other_dofs, other_dofs] *= self.OTHER_JOINT_COST
        Qpos = self.BALANCE_COST * Qbalance + Qjoint
        Qvel = 0.0 * np.eye(self.nv)
        Q = np.block([[Qpos, np.zeros((self.nv, self.nv))],
                      [np.zeros((self.nv,self.nv)), Qvel]])

        return Q

    def calc_costs(self):
        # Create a separate function for Q.
        R = self.set_R()
        Q = self.set_Q()

        # Allocate the A and B matrices. Mujoco does this analytically.
        A = np.zeros((2 * self.nv, 2 * self.nv))
        B = np.zeros((2 * self.nv, self.nu))
        flg_centered = True
        mujoco.mjd_transitionFD(self.model, self.data, self.epsilon, flg_centered, A, B, None, None)

        # Solve discrete Riccati equation.
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)

        # Compute the feedback gain matrix K.
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

