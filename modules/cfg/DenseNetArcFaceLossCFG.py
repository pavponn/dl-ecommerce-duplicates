import numpy as np

class DenseNetArcFaceLossCFG:
	image_size = 512
	batch_size = 16
	n_worker = 4
	init_lr = 3e-4
	dropout = 0.5
	n_epochs = 25
	n_splits = 5
	n_samples = None
	fold_id = 0
	valid_every = 1
	save_after = 0
	scale = 10
	margin = 0.5
	easy_margin = False
	search_space = np.arange(40, 100, 10)
	debug = False
	kernel_type = 'baseline'
	model_dir = './weights/'
	data_dir = './data/train_images'
	csv = './data/train.csv'