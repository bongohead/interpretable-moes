def get_linear_decay_coef(step: int, decay_steps: int, start_coef: float, end_coef: float) -> float:
    """
    Returns the current value for a coefficient that linearly decays from start_coef to end_coef between 0 and decay_steps, then 
     returns end_coef forever.

    Params:
        @step: The current step index
        @decay_steps: The point at which we return end_coef
        @start_coef: The starting coefficient value
        @end_coef: The end coefficient value
    """
    if step >= decay_steps:
        return end_coef
    frac = step / decay_steps
    return start_coef + frac * (end_coef - start_coef)