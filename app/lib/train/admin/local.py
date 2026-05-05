class EnvironmentSettings:
    def __init__(self):
        # === Kaggle Paths (writable) ===
        self.workspace_dir       = '/kaggle/working/output'
        self.save_dir            = '/kaggle/working/output'
        self.tensorboard_dir     = '/kaggle/working/output/tensorboard'
        self.pretrained_networks = '/kaggle/working/output/pretrained_networks'

        # === Aerial tracking dataset (input, read-only) ===
        # JPEG root — the *training source of truth*.
        # Contains: dataset{1..5}/<seq>/000000.jpg, annotation.txt, and
        #           metadata/contestant_manifest.json
        # AerialTracking reads from settings.env.aerial_dir, so this MUST
        # point at the JPEG mirror, not the original .mp4 dataset.
        self.aerial_dir            = '/kaggle/input/datasets/gamalasran/mtc-aic4-jpegs/mtc-aic4-jpegs'
        # Backward-compat alias (any older code referencing .frames_dir)
        self.frames_dir            = self.aerial_dir
        # Original .mp4 videos + sample_submission.csv live here
        self.aerial_videos_dir     = '/kaggle/input/datasets/gamalasran/aic-mtc-4'
        # Manifest path (also discoverable under aerial_dir/metadata/)
        self.manifest_path         = (
            '/kaggle/input/datasets/gamalasran/mtc-aic4-jpegs/'
            'mtc-aic4-jpegs/metadata/contestant_manifest.json'
        )
        # sample_submission.csv lives only in the videos dataset
        self.aerial_submission_csv = (
            '/kaggle/input/datasets/gamalasran/aic-mtc-4/'
            'metadata/sample_submission.csv'
        )

        # === Standard tracking dataset placeholders (unused, prevent KeyError) ===
        self.lasot_dir            = ''
        self.got10k_dir           = ''
        self.got10k_val_dir       = ''
        self.lasot_lmdb_dir       = ''
        self.got10k_lmdb_dir      = ''
        self.trackingnet_dir      = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir             = ''
        self.coco_lmdb_dir        = ''
        self.imagenet_dir         = ''
        self.imagenet_lmdb_dir    = ''
        self.imagenetdet_dir      = ''