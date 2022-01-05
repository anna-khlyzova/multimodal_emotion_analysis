from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import RandomSampler, SequentialSampler, random_split
from torchvision import transforms, models
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import time
import html
import emoji


"""This class takes care of tokenizing texts of posts.
   Input: texts.
   Output: input ids, attention masks."""
class RTokenizer():
	# first find the max len to pad or truncate to (this is done to make sure the texts are the same length,
    # otherwise, the model will throw an error)
    def __init__(self, texts):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True) 
        self.max_len = 0
        for text in texts:
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            self.max_len = max(self.max_len, len(input_ids))

    # tokenize text using max length
    def tokenize(self, text):
        encoded_dict = self.tokenizer.encode_plus(
                          text,
                          add_special_tokens = True,
                          max_length = self.max_len, # this is where the max length is used
                          padding="max_length",
                          truncation = True, 
                          return_attention_mask = True,
                          return_tensors = "pt")

        return encoded_dict['input_ids'][0], encoded_dict['attention_mask'][0]
    

class CustomDataset(Dataset):
    def __init__(self, txt, img, emo, rel, stim, tokenizer, img_dir):
        self.text, self. mask = self.tokenize_text(tokenizer, txt)
        self.image = self.transform_images(img, img_dir)
        self.emotion = emo
        self.relation = rel
        self.stimulus = stim
        
    def __len__(self):
        return len(self.emotion)
        
    def __getitem__(self, idx):
        text = self.text[idx]
        mask = self.mask[idx]
        image = self.image[idx]
        emotion = self.emotion[idx]
        relation = self.relation[idx]
        stimulus = self.stimulus[idx]
        return text, mask, image, emotion, relation, stimulus
    
    """ Preprocesses images into tensors.
    	Input: images, images directory.
    	Output: list of tensors for images."""
    def transform_images(self, images, dirr):
        transform = transforms.Compose([
            transforms.Resize((224,224)), # resize images to the same dimensions
            transforms.ToTensor(), # convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalize
        ])
        out_images = []
        for img in images:
            try:
                input_image = Image.open(dirr+"/"+img).convert("RGB") # open and convert image to RGB
                image = transform(input_image) # normalize and transform to tensor
                out_images.append(image)
            except Exception as e:
                print(e)
        return out_images
    
    """ Tokenizes texts.
    	Input: texts.
    	Output: input ids, masks."""
    def tokenize_text(self, tokenizer, texts):
        out_texts = []
        out_masks = []
        for txt in texts:
            txt = emoji.demojize(html.unescape(txt)) # transform emoji to text
            text, mask = tokenizer.tokenize(txt) # tokenize according to tokenizer class
            out_texts.append(text)
            out_masks.append(mask)
        return out_texts, out_masks

"""Text-based model using RoBERTa with three parallel layers (multitask) for emotion, 
	relation, and stimulus prediction."""
class TextModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(TextModel,self).__init__()
        self.base = RobertaModel.from_pretrained("roberta-base") # pretrained model
        self.fc1 = torch.nn.Linear(768, num_classes[0]) # for emotion classes
        self.fc2 = torch.nn.Linear(768, num_classes[1]) # for relation classes
        self.fc3 = torch.nn.Linear(768, num_classes[2]) # for stimulus classes

    def forward(self, ids, masks):
        logits = self.base(ids, masks)[1] # get pooler layer from roberta
        emotion = torch.sigmoid(self.fc1(logits)) # pass through fc and activation
        relation = torch.sigmoid(self.fc2(logits))
        stimulus = torch.sigmoid(self.fc3(logits))

        return [emotion, relation, stimulus]

"""Image-based model using ResNet with three parallel layers (multitask) for emotion, 
	relation, and stimulus prediction."""
class ImageModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ImageModel,self).__init__()
        self.base = models.resnet50(pretrained=True) # pretrained model
        self.base.fc = torch.nn.Linear(2048,2048) # change last fc layer of ResNet
        for name, param in self.base.named_parameters(): # freeze layers of preatrained model
            if name not in ["fc.weight","fc.bias"]:
                param.requires_grad = False
        self.fc1 = torch.nn.Linear(2048, num_classes[0]) # for emotion classes
        self.fc2 = torch.nn.Linear(2048, num_classes[1]) # for relation classes
        self.fc3 = torch.nn.Linear(2048, num_classes[2]) # for stimulus classes

    def forward(self,x):
        logits = self.base(x) # output of last ResNet layer
        emotion = torch.sigmoid(self.fc1(logits)) # pass through fc and activation
        relation = torch.sigmoid(self.fc2(logits))
        stimulus = torch.sigmoid(self.fc3(logits))

        return [emotion, relation, stimulus]

""" Late fusion model that uses trained-by-us text and image models.
    Input: text model, image model, classes. """
class LateModel(torch.nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super(LateModel,self).__init__()
        self.text_model = text_model # initiate base models
        self.image_model = image_model
        for param in self.image_model.parameters(): # freeze base models params to prevent retraining
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        self.image_model.eval()
        self.fc1 = torch.nn.Linear(16, num_classes[0]) # for emotions
        self.fc2 = torch.nn.Linear(10, num_classes[1]) # for relations
        self.fc3 = torch.nn.Linear(20, num_classes[2]) # for stimuli

    def forward(self,text,mask,image):
        out_text_emo = self.text_model(text,mask)[0] # pass for emo through text-based model ==> 8 classes
        out_image_emo = self.image_model(image)[0] # pass for emo through image-based model ==> 8 classes
        out_text_rel = self.text_model(text,mask)[1] # pass for rel through text-based model ==> 5 classes
        out_image_rel = self.image_model(image)[1] # pass for rel through image-based model ==> 5 classes
        out_text_stim = self.text_model(text,mask)[2] # pass for stim through text-based model ==> 10 classes
        out_image_stim = self.image_model(image)[2] # pass for stim through image-based model ==> 10 classes
        combined_emo = torch.cat([out_image_emo, out_text_emo], dim=1) # concatenate text and image features
        combined_rel = torch.cat([out_image_rel, out_text_rel], dim=1)
        combined_stim = torch.cat([out_image_stim, out_text_stim], dim=1)
        emotion = torch.sigmoid(self.fc1(combined_emo)) # pass through fc and activation
        relation = torch.sigmoid(self.fc2(combined_rel))
        stimulus = torch.sigmoid(self.fc3(combined_stim))

        return [emotion, relation, stimulus]

""" Early fusion model. No pre-trained models involved. Pure features combined."""
class EarlyModel(torch.nn.Module):
    def __init__(self, train_dataset, num_classes):
        super(EarlyModel,self).__init__()
        # combine features extracted from tokenizer and text preprocessing to calculate the dimensions
        num_features = train_dataset[0][0].size()[0]+train_dataset[0][2].flatten().size()[0]
        self.linear1 = torch.nn.Linear(num_features, 1024) # first linear layer
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc1 = torch.nn.Linear(256, num_classes[0]) # for emotions
        self.fc2 = torch.nn.Linear(256, num_classes[1]) # for relations
        self.fc3 = torch.nn.Linear(256, num_classes[2]) # for stimuli

    def forward(self,text,mask,image):
        combined = torch.cat([text, image.flatten(start_dim=1)], dim=1) # combine text and image
        out = self.linear1(combined)
        out = self.dropout(out)
        out = torch.nn.functional.relu(self.linear2(out))
        out = torch.nn.functional.relu(self.linear3(out))
        emotion = torch.sigmoid(self.fc1(out)) # pass through fc and activation
        relation = torch.sigmoid(self.fc2(out))
        stimulus = torch.sigmoid(self.fc3(out))

        return [emotion, relation, stimulus]
    
""" Model-based model. Uses models that are pretrained but not by us -- just pure roberta and resnet."""
class ModelBasedModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(ModelBasedModel,self).__init__()
        self.text_model = RobertaModel.from_pretrained('roberta-base') # text-based model
        self.image_model = models.resnet50(pretrained=True) # image-based model
        for name, param in self.image_model.named_parameters(): # freeze almost all resnet params
            if name not in ["fc.weight","fc.bias"]:
                param.requires_grad = False
        self.fc1 = torch.nn.Linear(1768, num_classes[0]) # for emotions
        self.fc2 = torch.nn.Linear(1768, num_classes[1]) # for relations
        self.fc3 = torch.nn.Linear(1768, num_classes[2]) # for stimuli

    def forward(self,text,mask,image):
        out_text = self.text_model(input_ids=text,attention_mask=mask)[1] # get output from pooler from roberta
        out_image = self.image_model(image) # get output from resnet
        combined = torch.cat([out_image, out_text], dim=1) # concat (you can change it to a more advanced combination)
        emotion = torch.sigmoid(self.fc1(combined)) # pass through fc and activation
        relation = torch.sigmoid(self.fc2(combined))
        stimulus = torch.sigmoid(self.fc3(combined))

        return [emotion, relation, stimulus]

""" Function to train the model.
	Input: model, optimizer, scheduler, dataloaders, optional: epochs, patience, mode.
	Output: model. """
def train(model, optimizer, scheduler, dataloaders_dict, epochs=25, patience=3, mode="model"):
    print("*** Start training ***")
    start_time = time.time()
    best_val_loss = np.inf # for early stopping
    wait = 0
    break_loop = False

    for epoch_i in range(0, epochs):
        epoch_start_time = time.time()
        if break_loop: # early stop if validation loss does not improve
            break
        print("\nTraining for epoch # {}.".format(epoch_i))
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                if epoch_i != 0: # step scheduler after the first epoch, otherwise a warning
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 # initialize current loss to 0 in every iteration
            preds_emo, preds_rel, preds_stim = [], [], [] # lists for predicted labels
            true_emo, true_rel, true_stim = [], [], [] # lists for actual labels

            for texts,masks,images,emo,rel,stim in dataloaders_dict[phase]:
                if mode != "image": # if it's any model but image (text or multimodal)
                    texts = texts.to("cuda")
                    masks = masks.to("cuda")
                if mode != "text": # if it is any model but text (image or multimodal)
                    images = images.to('cuda')
                # labels
                emo = emo.to("cuda")
                rel = rel.to("cuda")
                stim = stim.to("cuda")

                # Always clear any previously calculated gradients before performing a backward pass.
                optimizer.zero_grad()       
                # Perform a forward pass (evaluate the model on this training batch)
                with torch.set_grad_enabled(phase == "train"):
                    if mode == "text": # if it's the text model, send texts and masks into the model
                        logits = model(texts, masks)
                    elif mode == "image": # if it's the image model, send images
                        logits = model(images)
                    else: # otherwise send all
                        logits = model(texts, masks, images)

                	# compute losses for all three labels
                    loss_emo = torch.nn.BCELoss()(logits[0],emo.float())
                    loss_rel = torch.nn.BCELoss()(logits[1],rel.float())
                    loss_stim = torch.nn.BCELoss()(logits[2],stim.float())
                    loss = loss_emo + loss_rel + loss_stim # general loss

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                preds_emo.append(logits[0].cpu().detach().numpy().round())
                preds_rel.append(logits[1].cpu().detach().numpy().round())
                preds_stim.append(logits[2].cpu().detach().numpy().round())
                true_emo.append(emo.cpu().detach().numpy())
                true_rel.append(rel.cpu().detach().numpy())
                true_stim.append(stim.cpu().detach().numpy())

            # loss in each epoch, f-scores for all labels
            epoch_loss = running_loss / len(dataloaders_dict[phase])
            f1_emo = f1_score(np.concatenate(true_emo, axis=0), np.concatenate(preds_emo, axis=0), average="weighted")
            f1_rel = f1_score(np.concatenate(true_rel, axis=0), np.concatenate(preds_rel, axis=0), average="weighted")
            f1_stim = f1_score(np.concatenate(true_stim, axis=0), np.concatenate(preds_stim, axis=0), average="weighted")

            print("{} loss in epoch # {}: {}".format(phase, epoch_i,round(epoch_loss,3)))

            if phase == "val":
                print("emo_f1: {:.3f} *** rel_f1: {:.3f} *** stim_f1: {:.3f}".format(f1_emo,f1_rel,f1_stim))

                # early stopping
                if best_val_loss - epoch_loss > 0.005: # if the epoch is better, set the weight to 0 and save the state
                    best_val_loss = epoch_loss
                    wait = 0
                    model_state = model.state_dict()
                else: # otherwise wait untill the patience limit and restore
                    wait += 1
                    if wait >= patience:
                        print("Restoring model weights from the end of the best epoch.")
                        model.load_state_dict(model_state)
                        break_loop = True
        print("Finished training epoch # {} in {} seconds.".format(epoch_i, round((time.time() - epoch_start_time))))
    print("Training complete!")
    print("Finished training in {} seconds.".format(round((time.time() - start_time))))
    return model

""" Evaluate model.
	Input: model, dataloader, optional: mode.
	Output: classification reports with scores. """
def evaluate(model, test_dataloader, column_names, mode="model"):
    print("*** Evaluating the model ***")
    model.eval()
    results_emo, results_rel, results_stim = [], [], []
    trues_emo, trues_rel, trues_stim = [], [], []
    for texts,masks,images,emo,rel,stim in test_dataloader:
        if mode != "image": # if it's any model but image
            txt = texts.to('cuda')
            msk = masks.to('cuda')
        if mode != "text": # if it is any model but text
            img = images.to('cuda')
            
        if mode == "text": # if it's the text model, send texts and masks into the model
            result = model(txt, msk)
        elif mode == "image": # if it's the image model, send images
            result = model(img)
        else: # otherwise send all
            result = model(txt, msk, img)
        # move the results and true values to cpu
        results_emo.append(result[0].cpu().detach().numpy().round())
        results_rel.append(result[1].cpu().detach().numpy().round())
        results_stim.append(result[2].cpu().detach().numpy().round())
        trues_emo.append(emo.cpu().detach().numpy())
        trues_rel.append(rel.cpu().detach().numpy())
        trues_stim.append(stim.cpu().detach().numpy())

    # concatenate the results and true values
    results_emo = np.concatenate(results_emo, axis=0)
    trues_emo = np.concatenate(trues_emo, axis=0)
    results_rel = np.concatenate(results_rel, axis=0)
    trues_rel = np.concatenate(trues_rel, axis=0)
    results_stim = np.concatenate(results_stim, axis=0)
    trues_stim = np.concatenate(trues_stim, axis=0)
    # compare the results and true values
    results_emo = classification_report(trues_emo, results_emo, zero_division=0, output_dict=True, target_names=column_names[0])
    results_rel = classification_report(trues_rel, results_rel, zero_division=0, output_dict=True, target_names=column_names[1])
    results_stim = classification_report(trues_stim, results_stim, zero_division=0, output_dict=True, target_names=column_names[2])
    print("Weighted F1 for *** EMO: {} *** REL: {} *** STIM: {}".format(results_emo["weighted avg"]["f1-score"], 
    																	results_rel["weighted avg"]["f1-score"], 
    																	results_stim["weighted avg"]["f1-score"]))
    print("*** Finished evaluating ***")
    return results_emo, results_rel, results_stim

""" Shuffle dataset. Needed if traning model several times with a small dataset. """
def shuffle_dataset(train_dataset, test_dataset, batch_size=32):
    train_length = int(0.9*len(train_dataset))
    val_length = len(train_dataset) - train_length
    train_dataset, val_dataset = random_split(train_dataset, [train_length, val_length]) # split on train and val sets
    print("{0} training instances, {1} validation instances, {2} test instances.".format(train_length, 
    																					val_length, 
    																					len(test_dataset)))
    
    # push to dataloaders
    train_dataloader = DataLoader(train_dataset, 
                                  sampler = RandomSampler(train_dataset), 
                                  batch_size = batch_size)
    val_dataloader = DataLoader(val_dataset,
                            	sampler = SequentialSampler(val_dataset),
                            	batch_size = batch_size)
    test_dataloader = DataLoader(test_dataset,
                            	sampler = SequentialSampler(test_dataset),
                            	batch_size = batch_size)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    return dataloaders_dict, test_dataloader

""" Save results to a file and return them. """
def save_results(results, saveto):
    results_df = pd.DataFrame(columns=pd.DataFrame(results[0]).transpose().columns)
    for i in range(len(results)):
        df = pd.DataFrame(results[i]).transpose()
        df = df.round(2)
        label = df.iloc[0].name.split(".")[0]
        df.drop(["micro avg","samples avg", "macro avg"], axis=0, inplace=True)
        df.rename(index={"weighted avg": label+".weighted"}, inplace=True)
        results_df = results_df.append(df)
    results_df.index.name = "label"
    results_df.groupby("label").agg('mean').round(2).to_csv(saveto) # save to file
    return results_df.groupby("label").agg('mean').round(2)

""" Train and test data one or multiple times.
	Returns the model and the saved-to-a-file results. """
def train_and_test(mode, lrmain, lrlast, saveto, PATH, train_dataset, test_dataset, column_names, models=None, train_times=1, patience=3, epochs=20):
    results_list = []
    best_performance = 0.0
    # train for one or multiple iterations
    for i in range(train_times):

        dataloaders_dict, test_dataloader = shuffle_dataset(train_dataset, test_dataset) # shuffle before training and create dataloaders
        # choose the right model to instantiate
        if mode == "text":
            model = TextModel([len(column_names[0]), len(column_names[1]), len(column_names[2])])
        elif mode == "image":
            model = ImageModel([len(column_names[0]), len(column_names[1]), len(column_names[2])])
        elif mode == "early":
            model = EarlyModel(train_dataset, [len(column_names[0]), len(column_names[1]), len(column_names[2])])
        elif mode == "late":
            model = LateModel(models[0], models[1], [len(column_names[0]), len(column_names[1]), len(column_names[2])])
        elif mode =="model":
            model = ModelBasedModel([len(column_names[0]), len(column_names[1]), len(column_names[2])])
        else:
            print("Wrong model name. Use text, image, early, late, or model.")
            exit(-1)
            
        model.to('cuda')
        # choose optimizer with appropriate learning rates for different layers
        # lrmain is the main lr, lrlast is the lr for the top fc layer
        if mode == "text" or mode == "image":
            optimizer = optim.AdamW([
                    {"params":model.base.parameters(), "lr": lrmain},
                    {"params":model.fc1.parameters(), "lr": lrlast},
                    {"params":model.fc2.parameters(), "lr": lrlast},
                    {"params":model.fc3.parameters(), "lr": lrlast},

                    ],
                lr=lrmain)
        elif mode == "late":
            optimizer = optim.AdamW([
                    {"params":model.text_model.parameters()},
                    {"params":model.image_model.parameters()},
                    {"params":model.fc1.parameters(), "lr": lrlast},
                    {"params":model.fc2.parameters(), "lr": lrlast},
                    {"params":model.fc3.parameters(), "lr": lrlast},
                    ],
                lr=lrmain)
        elif mode == "early":
            optimizer = optim.AdamW(model.parameters(), lr=lrmain)
            
        elif mode == "model":
            optimizer = optim.AdamW([
                    {"params":model.text_model.parameters(), "lr": 3e-5},
                    {"params":model.image_model.parameters(), "lr": lrlast},
                    {"params":model.fc1.parameters(), "lr": lrlast},
                    {"params":model.fc2.parameters(), "lr": lrlast},
                    {"params":model.fc3.parameters(), "lr": lrlast},

                    ],
                lr=lrmain)

        # Decay LR by a factor of 0.1 every 10 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        model_ft = train(model=model,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         dataloaders_dict=dataloaders_dict, 
                         epochs=epochs, 
                         patience=patience, 
                         mode=mode)
        model_results = evaluate(model_ft, test_dataloader, column_names, mode=mode)
        # save the model if the results are better than the previous model
        new_results = model_results[0]["weighted avg"]["f1-score"] + model_results[1]["weighted avg"]["f1-score"] + model_results[2]["weighted avg"]["f1-score"]
        if new_results > best_performance:
            best_performance = new_results
            torch.save(model_ft.state_dict(), PATH)
            print("Saved the new best weights!")
        torch.cuda.empty_cache()
        del model_ft
        for res in model_results:
            results_list.append(res)
    saved_results = save_results(results_list, saveto) # saves and returns the results
    model.load_state_dict(torch.load(PATH)) # load the last best model weights if using model later
    return model, saved_results