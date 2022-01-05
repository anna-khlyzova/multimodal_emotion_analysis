import argparse
import os
import pandas as pd
import torch
import utils

def main():
	# load train and test data
	try:
		train_data = pd.read_csv(os.path.join(args.data_folder, "train_data.csv"))
		test_data = pd.read_csv(os.path.join(args.data_folder, "test_data.csv"))
	except Exception as e:
		print(e)
		exit(-1)

	# check if image directory exist
	img_dir = os.path.join(args.data_folder, "images")
	if os.path.isdir(img_dir) == False:
		print("Incorrect image directory {}.".format(img_dir))
		exit(-1)
	# extract columns for three labels
	emo_columns = [col for col in train_data.columns if col.startswith("emotion")]
	rel_columns = [col for col in train_data.columns if col.startswith("relation")]
	stim_columns = [col for col in train_data.columns if col.startswith("stimulus")]

	# create datasets
	tokenizer = utils.RTokenizer(train_data["text"])
	train_dataset = utils.CustomDataset(train_data["text"], train_data["image"], train_data[emo_columns].values, train_data[rel_columns].values, train_data[stim_columns].values, tokenizer, img_dir)
	test_dataset = utils.CustomDataset(test_data["text"], test_data["image"], test_data[emo_columns].values, test_data[rel_columns].values, test_data[stim_columns].values, tokenizer, img_dir)
	
	# check if directory for results exist
	if os.path.isdir("./results") == False:
		os.mkdir("./results")
		print("Created results directory.")
	# check if directory for weights exist
	if os.path.isdir("./weights") == False:
		os.mkdir("./weights")
		print("Created weights directory.")

	trained_models = None
	print(args.model + ' model...')
	if args.model == "text":
		lrmain = 3e-5 # change if needed (hyperparam)
		lrlast = 3e-3 # change if needed (hyperparam)
	elif args.model == "image":
		lrmain = 0.0001 # change if needed (hyperparam)
		lrlast = 0.001 # change if needed (hyperparam)
	elif args.model == "early":
		lrmain = 0.0001 # change if needed (hyperparam)
		lrlast = 0.001 # change if needed (hyperparam)
	elif args.model == "late":
		text_model = utils.TextModel([len(emo_columns), len(rel_columns), len(stim_columns)])
		text_model.to("cuda")
		text_model.load_state_dict(torch.load("./weights/text_model_weights.pth"))
		image_model = utils.ImageModel([len(emo_columns), len(rel_columns), len(stim_columns)])
		image_model.to("cuda")
		image_model.load_state_dict(torch.load("./weights/image_model_weights.pth"))
		trained_models = (text_model, image_model)
		lrmain = 0.001 # change if needed (hyperparam)
		lrlast = 0.01 # change if needed (hyperparam)
	elif args.model == "model":
		lrmain = 0.0001 # change if needed (hyperparam)
		lrlast = 0.001 # change if needed (hyperparam)
	model, results = utils.train_and_test(args.model, 
											lrmain, 
											lrlast, 
											"./results/" + args.model + "_results.csv", 
											"./weights/" + args.model + "_model_weights.pth",
											train_dataset,
											test_dataset,
											(emo_columns, 
											rel_columns, 
											stim_columns),
											models = trained_models,
											train_times=1) # change train times if needed
	print(results)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data_folder", type=str, help="data folder", nargs='?', default="data")
	parser.add_argument("model", type=str, help="type of model", choices=["text","image","early","late","model"], nargs='?', default="model")
	args = parser.parse_args()
	main()