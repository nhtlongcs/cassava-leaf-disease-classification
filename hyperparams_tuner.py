import sys

sys.path.append("./libs/")

import argparse
import pprint

import torch
import torch.nn as nn
import yaml
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from utils.getter import get_instance
from utils.random_seed import set_seed
from workers.trainer import Trainer

import optuna
from torch.optim import SGD, Adam, RMSprop

def train(trial, config):
	assert config is not None, "Do not have config file!"

	# pprint.PrettyPrinter(indent=2).pprint(config)

	dev_id = (
		"cuda:{}".format(config["gpus"])
		if torch.cuda.is_available() and config.get("gpus", None) is not None
		else "cpu"
	)
	device = torch.device(dev_id)

	# Get pretrained model
	pretrained_path = config["pretrained"]

	pretrained = None
	if pretrained_path != None:
		pretrained = torch.load(pretrained_path, map_location=dev_id)
		for item in ["model"]:
			config[item] = pretrained["config"][item]

	# 1: Load datasets
	set_seed()
	train_dataset = get_instance(config["dataset"]["train"])
	train_dataloader = get_instance(
		config["dataset"]["train"]["loader"], dataset=train_dataset
	)

	set_seed()
	val_dataset = get_instance(config["dataset"]["val"])
	val_dataloader = get_instance(
		config["dataset"]["val"]["loader"], dataset=val_dataset
	)

	# 2: Define network
	set_seed()
	model = get_instance(config["model"]).to(device)

	# Train from pretrained if it is not None
	if pretrained is not None:
		model.load_state_dict(pretrained["model_state_dict"])

	# 3: Define loss
	set_seed()
	criterion = get_instance(config["loss"]).to(device)

	# 4: Define Optimizer
	set_seed()
	optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
	lr = trial.suggest_float("lr", low=0.0001, high=0.01, log=True)
	optimizer = getattr(torch.optim, optimizer_name)(lr=lr, params=model.parameters()) 

	if pretrained is not None:
		optimizer.load_state_dict(pretrained["optimizer_state_dict"])

	# 5: Define Scheduler
	set_seed()
	scheduler = get_instance(config["scheduler"], optimizer=optimizer)

	# 6: Define metrics
	set_seed()
	metric = {mcfg["name"]: get_instance(mcfg) for mcfg in config["metric"]}

	if config["fp16"]:
		model, optimizer = amp.initialize(
			models=model, optimizers=optimizer, opt_level=config["fp16_opt_level"]
		)
		amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

	# 7: Create trainer
	set_seed()
	trainer = Trainer(
		device=device,
		config=config,
		model=model,
		criterion=criterion,
		optimier=optimizer,
		scheduler=scheduler,
		metric=metric,
	)

	# 8: Start to train
	set_seed()
	trainer.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader)
	return trainer.best_metric[config["metric"][0]["name"]].item()

class Objective(object):
	def __init__(self, config):
		self.config = config
	
	def __call__(self, trial):
		val_acc = train(trial, self.config)
		sys.stdout.flush()
		return val_acc


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config")
	parser.add_argument("--gpus", default=None)
	parser.add_argument("--fp16", default=True)
	parser.add_argument("--fp16_opt_level", default="O2")
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()
	config_path = args.config
	config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
	config["gpus"] = args.gpus
	config["debug"] = args.debug
	config["fp16"] = args.fp16 if str(args.fp16).lower() != "false" else False
	config["fp16_opt_level"] = args.fp16_opt_level


	study = optuna.create_study(direction="maximize")
	study.optimize(Objective(config), n_trials=1)
	print("Best trial:")
	trial = study.best_trial
	print("  Value: ", trial.value)
	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))