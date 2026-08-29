"""Runnable companion generated for the corresponding gp3ml 0.3.0 vignette."""
import gp3mlpy as gp

reg = gp.gp3ml_api_contracts()
assert len(reg.exports) == 127
assert gp.r_reference_version == "0.3.0"

data = gp.simulate_gazepoint_governed_data(n_participants=12, n_stimuli=4, trials_per_cell=1, seed=17)
task = gp.create_gazepoint_synthetic_task(data, workflow="assigned_condition", generalization_target="new_trials_known_participants")
assert task.generalization_target == "new_trials_known_participants"
