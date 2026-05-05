from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # === Kaggle Paths ===
    PROJECT_DIR = '/kaggle/working/TrackingMamba'
    OUTPUT_DIR  = '/kaggle/working/output'

    settings.prj_dir              = PROJECT_DIR
    settings.save_dir             = OUTPUT_DIR
    settings.network_path         = f'{OUTPUT_DIR}/test/networks'
    settings.result_plot_path     = f'{OUTPUT_DIR}/test/result_plots'
    settings.results_path         = f'{OUTPUT_DIR}/test/tracking_results'
    settings.segmentation_path    = f'{OUTPUT_DIR}/test/segmentation_results'

    # === Aerial dataset paths (JPEG root is source of truth) ===
    # Mirror the train env naming so test-time code can use either.
    settings.aerial_dir            = '/kaggle/input/datasets/gamalasran/mtc-aic4-jpegs/mtc-aic4-jpegs'
    settings.aerial_path           = settings.aerial_dir   # legacy alias (was videos root — repointed)
    settings.frames_path           = settings.aerial_dir   # legacy alias
    settings.aerial_videos_dir     = '/kaggle/input/datasets/gamalasran/aic-mtc-4'
    settings.manifest_path         = (
        '/kaggle/input/datasets/gamalasran/mtc-aic4-jpegs/'
        'mtc-aic4-jpegs/metadata/contestant_manifest.json'
    )
    settings.aerial_submission_csv = (
        '/kaggle/input/datasets/gamalasran/aic-mtc-4/'
        'metadata/sample_submission.csv'
    )

    # === Placeholder paths (unused, prevent attribute errors) ===
    settings.got10k_path                      = ''
    settings.got10k_lmdb_path                 = ''
    settings.lasot_path                       = ''
    settings.lasot_lmdb_path                  = ''
    settings.lasot_extension_subset_path_path = ''
    settings.trackingnet_path                 = ''
    settings.nfs_path                         = ''
    settings.otb_path                         = ''
    settings.tc128_path                       = ''
    settings.tnl2k_path                       = ''
    settings.itb_path                         = ''
    settings.vot_path                         = ''
    settings.vot18_path                       = ''
    settings.vot22_path                       = ''
    settings.uav_path                         = ''
    settings.tpl_path                         = ''
    settings.davis_dir                        = ''
    settings.youtubevos_dir                   = ''
    settings.got_packed_results_path          = ''
    settings.got_reports_path                 = ''
    settings.tn_packed_results_path           = ''

    return settings