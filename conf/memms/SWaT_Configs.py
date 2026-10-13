class Config:
    """Independent SWaT experiment configuration."""

    def __init__(self):
        self.dataset = 'SWaT'
        self.input_channels = 51

        # Encoder and experts.
        self.window_size = 258
        self.time_step = 258
        self.d_model = 32
        self.tcn_layers = 2
        self.dropout = 0.1
        self.num_experts = 2

        # Memory and the two combined objectives.
        self.normal_memory_size = 32
        self.memory_topk = 8
        self.memory_strength = 0.6
        self.lambda_aux = 0.1
        self.score_memory_weight = 1.0

        # Training and validation.
        self.num_epoch = 10
        self.early_stop = 5
        self.batch_size = 128
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.num_workers = 0
        self.drop_last = False
        self.study_name = "final"
        self.trial_id = "final"
        self.lr = 0.0003
        self.weight = 0.0005

        # Anomaly Detection parameters
        self.detect_nu = 0.0005
        self.threshold_determine = 'floating'

        # Output and profiling options.
        self.profile_timing = False  # Record training and evaluation timing.
        self.save_diagnostics = True  # Save per-window scores and evaluation diagnostics.
        self.keep_checkpoints = True  # Retain the calibrated best model after evaluation.
