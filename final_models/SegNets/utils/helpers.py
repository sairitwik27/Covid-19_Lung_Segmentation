import torch
from dataset import BinaryCovid
from torch.utils.data import DataLoader

def save_checkpoint(state,filename="Ritwik_checkpoint.pth.tar"):
  torch.save(state,filename)


def load_checkpoint(checkpoint,model):
  print("Loading")
  model.load_state_dict(checkpoint["state_dict"])

def load_data(train_img_dir,train_gt_dir,val_img_dir,val_gt_dir,batch_size,trainsize,num_workers=2,pin_memory=True):

  train_dataset = BinaryCovid(image_root = train_img_dir,
                              gt_root = train_gt_dir,
                              trainsize = trainsize,
                              )
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=pin_memory,
      shuffle=True
  )

  val_dataset = BinaryCovid(
      image_root = val_img_dir,
      gt_root = val_gt_dir,
      trainsize = trainsize,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=pin_memory,
      shuffle=False
  )

  return train_loader,val_loader

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def get_dice(loader,model,device="cuda"):
  dice_score = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y = y.to(device)
      pred = torch.sigmoid(model(x))
      pred = (pred>0.5).float()
      dice_score+=(2*(pred*y).sum())/((pred+y).sum()+1e-8)

  print(f"DICE SCORE: {dice_score/len(loader)}")
  model.train()

def get_IoU(loader,model,device="cuda"):
  IoU = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y = y.to(device)
      pred = torch.sigmoid(model(x))
      pred = (pred>0.5).float()
      IoU+= iou(pred,y)

  print(f"IoU SCORE: {IoU}")
  model.train()

def get_metrics(loader,model,device="cuda"):
  IoU = 0
  dice_score = 0
  f1_pos = 0
  f1_neg = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y = y.to(device)
      pred = torch.sigmoid(model(x))
      pred = (pred>0.5).float()
      dice_score+=dice(pred,y)
      IoU+= iou(pred,y)
      F1_pos,F1_neg=Fscore(pred,y)
      f1_pos+=F1_pos
      f1_neg+=F1_neg
      
  #print(f"IoU SCORE: {IoU/len(loader)}")
  print(f"Dice score: {dice_score/len(loader)}")
  print(f"Sensitivity score: {f1_pos/len(loader)}")
  print(f"Specificity score: {f1_neg/len(loader)}")
  model.train()
  
def iou(prediction: torch.Tensor, truth: torch.Tensor):
    true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,truth)
    iou = 1 - (true_positives/(true_positives+false_positives+false_negatives))
    
    return iou

def dice(outputs: torch.Tensor, labels: torch.Tensor):
    dice = (2*(outputs*labels).sum())/((outputs+labels).sum()+1e-8)
    
    return dice


def confusion(prediction, truth):
    confusion_vector = prediction / truth
    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def Fscore(prediction, truth):
    true_positives, false_positives, true_negatives, false_negatives = confusion(prediction,truth)
    print(true_positives, false_positives, true_negatives, false_negatives)
    precision_pos = true_positives/(true_positives+false_positives+1e-8)
    recall_pos = true_positives/(true_positives+false_negatives+1e-8)
    precision_neg = true_negatives/(true_negatives+false_negatives+1e-8)
    recall_neg = true_negatives/(true_negatives+false_positives+1e-8)
    #F1_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos+1e-8)
    #F1_neg = 2*precision_neg*recall_neg/(precision_neg+recall_neg+1e-8)
    F1_pos = true_positives/(true_positives+false_negatives)
    F1_neg = true_negatives/(true_negatives+false_positives)
    return F1_pos,F1_neg