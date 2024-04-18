import os

class config_siamese():
    def __init__(self):
        
        self.lr = 2e-3
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 100
        self.warmup_epochs = 10
        self.batch_size = 64
        self.clip_grad = 0.8
        self.use_time_cond = True
        self.accum_iter=1
        self.global_pool = False
        
        self.embedding_size = 512
        self.num_res = 4

        # Project setting
        self.root_path = '/media/arnav/DC12B70112B6E026/DreamDiffusion/DreamDiffusion'
        self.output_path = '/media/arnav/DC12B70112B6E026/DreamDiffusion/dreamdiffusion/exps_siamese'
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_55_95_std.pth')
        self.image_path = './DreamDiffusion/datasets/imageNet_images'
        self.neg_per_eeg = 10
        self.seed = 2022
        # self.roi = 'VC'
        # self.crop_ratio = 0.2
        # self.img_size = 512
        # self.aug_times = 1
        # self.num_sub_limit = None
        # self.include_hcp = True
        # self.include_kam = True
        # self.accum_iter = 1
        # self.HW = None
        # self.use_nature_img_loss = False
        # self.img_recon_weight = 0.5
        # self.focus_range = None # [0, 1500] # None to disable it
        # self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0
