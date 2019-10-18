'''
higher level operational utils
'''
import data
from torch.utils.data import DataLoader
from data.dataset import AdpDataset_3d
import torchvision.transforms as transforms

def genDsLoader(opts, mode=0):
	'''
	generate dataset, dataLoader and iterator list (singler)
	:param opts:
	:param mode: 0 for train , 1 for test, 2 for test in loop (with folding)
	:return: datasets_li, loader_li, iterator li
	'''
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=opts.pixel_mean, std=opts.pixel_std)])

	if 0 == mode:
		adpDs_li, loader_li, iter_li = [], [], []

		for i in range(len(opts.trainset)):
			ds = eval(opts.trainset[i])("train", opts=opts)
			ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, True, trans, opts=opts)
			loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus*opts.batch_size//len(opts.trainset), shuffle=True, num_workers=opts.n_thread, pin_memory=True)
			iterator = iter(loader)
			# to list
			adpDs_li.append(ds_adp)
			loader_li.append(loader)
			iter_li.append(iterator)
		return adpDs_li, loader_li, iter_li

	elif 1 == mode:
		ds = eval(opts.testset)("train", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts)
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=True, num_workers=opts.n_thread, pin_memory=True)
		iterator = iter(loader)

		return ds_adp, loader, iterator
	elif 2 == mode:
		ds = eval(opts.testset)("train", opts=opts)
		ds_adp = AdpDataset_3d(ds, opts.ref_joints_name, False, trans, opts=opts, n_folding=opts.n_foldingLpTest)   # few test in loop
		loader = DataLoader(dataset=ds_adp, batch_size=opts.num_gpus * opts.batch_size // len(opts.trainset),
		                    shuffle=True, num_workers=opts.n_thread, pin_memory=True)
		iterator = iter(loader)

		return ds_adp, loader, iterator
	else:
		print('no such mode defined')
		return -1