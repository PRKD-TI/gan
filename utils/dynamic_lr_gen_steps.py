# ------------------ adaptive generator steps ------------------
def get_dynamic_gen_steps(loss_D, max_steps=5):
    if loss_D > 1.0:
        return max_steps
    elif loss_D > 0.5:
        return 3
    elif loss_D > 0.2:
        return 2
    return 1
