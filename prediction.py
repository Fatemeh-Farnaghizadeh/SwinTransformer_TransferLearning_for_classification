import utils
import dataset
import torch
import pandas as pd
import os

from torch import nn
import torchvision


if __name__ == '__main__':

    # Load model structure
    model = torchvision.models.swin_t().to(utils.DEVICE)
    model_weights = torchvision.models.Swin_T_Weights.DEFAULT

    model.heads = nn.Linear(in_features=model.head.in_features, out_features=utils.NUM_CLASSES).to(utils.DEVICE)

    # Load your custom weights from a .pth file
    saved_model_path = os.path.join(utils.SAVE_MODEL_PATH, 'Swin_T_Model.pth')
    custom_weights = torch.load(saved_model_path)

    # Set the custom weights for the entire model
    model.load_state_dict(custom_weights)
    model.eval()

    # Load test_data & make predictions
    transformer = model_weights.transforms()

    test_data, _ = dataset.get_test_loader(transformer)

    for idx, (images, labels) in enumerate(test_data):

        if idx == 0:
            images = images.to(utils.DEVICE)
            labels = labels.to(utils.DEVICE)

            with torch.no_grad():
                preds = model(images)
        
            preds_labels = torch.argmax(preds, dim=1)
        
            for i in range(len(preds_labels)):
                print(f"Predicted: {preds_labels[i].item()}, Ground Truth: {labels[i].item()}")   

            df_results = pd.DataFrame({'preds': preds_labels.cpu(), 'labels': labels.cpu()})
            csv_filename = 'test_results.csv'  
            df_results.to_csv(csv_filename, index=False)  

        else:
            break  