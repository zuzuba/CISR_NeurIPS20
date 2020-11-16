from src.CMDP_solvers import LagrangianCMDPSolver

__all__ = ['LagrangianStudent']


class LagrangianStudent(LagrangianCMDPSolver):
    """
    A Student is a CMDP solver with additional functionality to transfer the solution of one CMDP to the next one
    proposed by the teacher.
    """
    def __init__(self, env, br_algo, online_algo, curriculum_transfer, br_kwargs=None, online_kwargs=None,
                 _init_setup_model=True, lagrangian_ronuds=10,
                 log_training=False, br_uses_vec_env=False, n_envs=1,
                 use_sub_proc_env=True):

        # Initialize the solver
        super().__init__(env, br_algo, online_algo, br_kwargs, online_kwargs,
                         _init_setup_model, lagrangian_ronuds, log_training,
                         br_uses_vec_env, n_envs, use_sub_proc_env)

        # Add the transfer capabilities
        self.curriculum_transfer = curriculum_transfer

    def set_env(self, env, keep_multipliers=False,
                reset_br=False, same_env=False,):
        # In case the teacher proposes twice the same env in a row we do not
        # reset the parameters
        if same_env:
            pass
        else:
            # We can only transfer is env had been previously initialized
            can_transfer = self._env is not None

            if can_transfer:
                old_params = self.get_params()

            super().set_env(env, keep_multipliers, reset_br)

            if can_transfer:
                self.set_params(self.curriculum_transfer(old_params))

