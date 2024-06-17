from models import *
from utils import *
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# Evaluate function
def evaluate(encoder, fc, generator, device):
    labels = np.arange(0, 13)
    Y = []
    Y_hat = []
    for x, y in generator:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = encoder(x)
        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "auc": auc,
    }
    # df_cm = pd.DataFrame(confusion_matrix(Y, Y_hat.argmax(axis=1)))
    return metrics


def train(args):
    subject = args.subject
    device = args.device
    device = torch.device(device)
    batch_size = 32
    batch_size2 = 260
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    root_dir = ""
    data_dir = ""
    data_folder = ""
    sessions = [""]
    model_dir = ""
    
    # Load data
    X, Y = load_data(root_dir=root_dir, data_dir = data_dir, data_folder = data_folder, subject=subject,  session=sessions[0])
    # Dataloader
    train_loader, test_loader = get_dataloader(
        X, Y, batch_size, batch_size2, seed, shuffle=True
    )

    # Define model
    num_classes = 13
    channels = X.shape[1]
    
    os.makedirs(f'./{num_classes}class/{model_dir}/{subject}', exist_ok=True)
    log_file = open(f'./{num_classes}class/{model_dir}/{subject}/{subject}_log.log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    print("Random Seed: ", seed)
    print("subject: ", subject)
    print("device: ", device)

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512

    ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(
        device
    )
    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    print("ddpm size: ", sum(p.numel() for p in ddpm.parameters()))
    print("encoder size: ", sum(p.numel() for p in encoder.parameters()))
    print("decoder size: ", sum(p.numel() for p in decoder.parameters()))
    print("fc size: ", sum(p.numel() for p in fc.parameters()))

    # Criterion
    criterion = nn.L1Loss()
    criterion_class = nn.MSELoss()

    # Define optimizer
    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

    # EMAs
    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10,)

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(
        optimizer=optim1,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    scheduler2 = optim.lr_scheduler.CyclicLR(
        optimizer=optim2,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    # Train & Evaluate
    num_epochs = 500
    test_period = 1
    start_test = test_period
    alpha = 0.1

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_auc = 0

    with tqdm(
        total=num_epochs, desc=f"Method ALL - Processing subject {subject}"
    ) as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            diffe.train()

            ############################## Train ###########################################
            for x, y in train_loader:
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                y_cat = F.one_hot(y, num_classes=num_classes).type(torch.FloatTensor).to(device)
                # Train DDPM
                optim1.zero_grad()
                x_hat, down, up, noise, t = ddpm(x)

                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()
                ddpm_out = x_hat, down, up, t

                # Train Diff-E
                optim2.zero_grad()
                decoder_out, fc_out = diffe(x, ddpm_out)

                loss_gap = criterion(decoder_out, loss_ddpm.detach())
                loss_c = criterion_class(fc_out, y_cat)
                loss = loss_gap + alpha * loss_c
                loss.backward()
                optim2.step()

                # Optimizer scheduler step
                scheduler1.step()
                scheduler2.step()

                # EMA update
                fc_ema.update()

            ############################## Test ###########################################
            with torch.no_grad():
                if epoch > start_test:
                    test_period = 1
                if epoch % test_period == 0:
                    ddpm.eval()
                    diffe.eval()

                    metrics_test = evaluate(diffe.encoder, fc_ema, test_loader, device)

                    acc = metrics_test["accuracy"]
                    f1 = metrics_test["f1"]
                    recall = metrics_test["recall"]
                    precision = metrics_test["precision"]
                    auc = metrics_test["auc"]

                    best_acc_bool = acc > best_acc
                    best_f1_bool = f1 > best_f1
                    best_recall_bool = recall > best_recall
                    best_precision_bool = precision > best_precision
                    best_auc_bool = auc > best_auc

                    if best_acc_bool:
                        best_acc = acc
                        torch.save(diffe.state_dict(), f'./{num_classes}class/{model_dir}/{subject}/{subject}_{best_acc*100:.2f}.pt')
                    if best_f1_bool:
                        best_f1 = f1
                    if best_recall_bool:
                        best_recall = recall
                    if best_precision_bool:
                        best_precision = precision
                    if best_auc_bool:
                        best_auc = auc
                    
                    description = f"Best accuracy: {best_acc*100:.2f}%"
                    pbar.set_description(
                        f"Method ALL - Processing subject {subject} - {description}"
                    )
                    print()
                    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")
            pbar.update(1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Specify the device here
    subjects = [1, 6]  # Specify the range of subjects here (1 to 10)
    
    for subject in range(subjects[0], subjects[1]+1):
        args = argparse.Namespace(device=device, subject=subject)
        train(args)