#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.

from args import get_parser
import torch
#from testing import *
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os, random, math
from models import fc_model
from dataset import get_loaderTrain, get_loaderValid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------
base_acc=0
def main(args): 
   
    data_loader, dataset = get_loaderTrain(args.data_dir, batch_size=args.batch_size, shuffle=True, 
                                           num_workers=args.num_workers, drop_last=False, args=args)
    data_loaderValid, datasetValid = get_loaderValid(args.data_dir, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=False, args=args)

    data_size = dataset.get_data_size()
    num_classes = dataset.get_num_classes()
    instance_size = dataset.get_instance_size()

    # Build the model
    model = fc_model(input_size=instance_size, num_classes=num_classes, dropout=args.dropout)

    # create optimizer
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.learning_rate)
    
    # multi-class hinge loss
    label_crit = nn.MultiMarginLoss(reduce=True)

    model = model.to(device)
    model.train()

    print ("model created & starting training ...\n\n")
    # Training script
    for epoch in range(args.num_epochs):

        total_correct_preds = 0.0
        total = 1e-10
        loss = 0.0
        
        # step loop
        for step, (image_input, class_idxs) in enumerate(data_loader):

            #print("The size of tensor: ",(image_input.size()))
            
            # move all data loaded from dataloader to gpu
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)

            # feed-forward data in the model
            output = model(image_input) # 32 * 150528 --> 32 * 11

            # compute losses
            state_loss = label_crit(output, class_idxs) # --> 32 * 1

            # aggregate loss for logging
            loss += state_loss.item()

            # back-propagate the loss in the model & optimize
            model.zero_grad()
            state_loss.backward()
            optimizer.step()

            # accuracy computation
            _, pred_idx = torch.max(output, dim=1)
            total_correct_preds += torch.sum(pred_idx==class_idxs).item()
            total += output.size(0)

        # epoch accuracy & loss
        accuracy = round(total_correct_preds/total, 2)
        loss = round(loss/total, 2)

	# you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

        print('\repoch {}: accuracy: {}, loss: {}'.format(epoch, accuracy, loss), end="")
        x = validate(model,data_loaderValid, datasetValid)
        save_model(model,epoch,optimizer,loss,x)
    print ()
    #validate(model)
    
    #if args.mode == "test":
      #load_checkpoint(checkpoint.pth)
      #testingacc(data_loader)



def validate(model,data_loaderValid, datasetValid):
      model = model.to(device)
      model.eval()
      total = 1e-10
      total_correct_predsValid = 0.0
      for i, (image_input,class_idxs) in enumerate(data_loaderValid):
          #print("-____-")
          class_idxs = class_idxs.to(device)
          image_input = image_input.to(device)
          outputs = model(image_input)
          _, pred_idx = torch.max(outputs, dim=1)
          total_correct_predsValid += torch.sum(pred_idx == class_idxs).item()
          total += outputs.size(0)
      #print('---',total_correct_predsValid)
      accuracyValid = round(total_correct_predsValid / total, 2)
      print(' validation accuracy:', accuracyValid)
      return accuracyValid
      

def save_model(model,epoch,optimizer,loss,accuracy):
  global base_acc
  if accuracy > base_acc:
      print('save checkpoint')
      torch.save(model,'model.pth')
      #torch.save(state,'checkpoint.pth')
      base_acc=accuracy
      #print(base_acc)
    

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
