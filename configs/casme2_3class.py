CONFIG = {
    "dataset": "casme2",

    "data": {
        "csv_path": "data/annotations/casme2_3class.csv",
        "rgb_root": "/root/lizhiqi/casme2_256",
        "flow_root": "/root/lizhiqi/CASME2/flow_strain",
    },

    "pretrained_weights": "/root/NFS_data/pretrain_pool/pretrain.pth",
    "output_root": "/root/NFS_data/results/casme2_3class",
    "gpu_id": "3",

    "train_ratio": 1.0,
    "val_batch_size": 256,
    "unfreeze_epoch": 100,

    "htnet": {
        "image_size": 28,
        "patch_size": 7,
        "dim": 256,
        "heads": 3,
        "num_hierarchies": 3,
        "block_repeats": (2, 2, 8),
    },

    "param_grid": {
        "learning_rate": [5e-5],
        "weight_decay": [1e-4],
        "batch_size": [128],
        "epochs": [500],
        "num_classes": [3],
        "seed": [42],
    },
}