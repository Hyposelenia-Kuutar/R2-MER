CONFIG = {
    # Datasets included in the unified Apex pool.
    "datasets": ["casme2", "smic", "samm"],

    "data": {
        "casme2": {
            "rgb_root": "/root/lizhiqi/casme2_256",
            "csv_path": "data/annotations/casme2_3class.csv",
        },

        "smic": {
            "rgb_root": "/root/lizhiqi/smic_256",
            "csv_path": "data/annotations/smic_3class.csv",
        },

        "samm": {
            "rgb_root": "/root/lizhiqi/samm_256",
            "csv_path": "data/annotations/samm_3class.csv",
        },
    },

    "mix_save_base": "/root/NFS_data/multi_exchange_pool",
    "save_path": "/root/NFS_data/pretrain_pool/pretrain.pth",

    # Visible GPUs for DataParallel pre-training.
    "gpu_ids": "0,1",

    "num_classes": 3,
    "epochs": 100,
    "batch_size": 128,

    "optimizer": {
        "lr": 5e-5,
        "weight_decay": 0.01,
    },

    "loss": {
        "lambda_l1": 0.6,
        "lambda_ssim": 0.4,
    },

    "grad_clip": 1.0,

    # None means no subject is excluded from the pre-training pool.
    "test_subject": None,
}