import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



import datetime

from RecFormer.RFPIOProject.Dataloader import MultiOmicsDataset
from RecFormer.RFPIOProject.models import MultiViewLoss, MultiViewPerceiver
from RecFormer.RFPIOProject.utils import getMvKNNGraph


cuda = torch.cuda.is_available()


class Trainer:
    def __init__(self, model, optimizer, dataset, loss_fn, device, batch_size=8, n_epochs=100):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.val_criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def train_1_epoch(self, train_loader, epoch, is_recon=True):
        self.model.train()
        total_loss = 0.0
        recon_loss = 0.0
        consist_loss = 0.0
        graph_loss = 0.0
        cls_loss = 0.0
        disentangle_loss = 0.0
        reconstructions_data = []

        for batch_idx, (data_list, labels, mask, cdata, feature_key) in enumerate(tqdm(train_loader, desc=f"Train 1 Epoch {epoch}")):

            data_list = [d.to(self.device) for d in data_list]
            mask = mask.to(self.device)
            labels = labels.to(self.device)


            self.optimizer.zero_grad()
            outputs = self.model(data_list, mask, is_recon=is_recon)


            reconstructions_data.append({
                'reconstructions': [rec.detach().clone() for rec in outputs['reconstructions']],
                'orig_data': [data.detach().clone() for data in cdata],
                'mask': mask.detach().clone(),
                'labels': labels,
            })


            with torch.no_grad():
                if cdata[0].shape[0] == self.batch_size:
                    k = 5
                else:
                    k = cdata[0].shape[0] - 1

                adj_matrix = getMvKNNGraph([c.cpu().numpy() for c in cdata], k=k)
                adj_matrix = torch.tensor(adj_matrix, device=self.device, dtype=torch.float32)

                reconstructions_adj_matrix = getMvKNNGraph([c.cpu().numpy() for c in outputs['reconstructions']], k=k)
                reconstructions_adj_matrix = torch.tensor(reconstructions_adj_matrix, device=self.device, dtype=torch.float32)

            loss_dict = self.loss_fn(
                preds=outputs['class_result'],
                labels=labels,
                row_data=cdata,
                reconstructions_data=outputs['reconstructions'],
                row_adj_matrix=adj_matrix,
                reconstructions_adj_matrix=reconstructions_adj_matrix,
                masks=mask,
                embeddings=outputs['view_embeddings'],
            )


            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()


            total_loss += loss_dict['total_loss'].item()
            recon_loss += loss_dict['recon_loss'].item()
            consist_loss += loss_dict['consist_loss'].item()
            graph_loss += loss_dict['graph_loss'].item()
            cls_loss += loss_dict['cls_loss'].item()
            disentangle_loss += loss_dict['disentangle_loss'].item()


        steps = len(train_loader)
        return {
            'total': total_loss / steps,
            'recon': recon_loss / steps,
            'consist': consist_loss / steps,
            'graph': graph_loss / steps,
            'cls': cls_loss / steps,
            'disentangle': disentangle_loss / steps,
            'reconstructions_data': reconstructions_data,
        }

    def train_2_epoch(self, train_loader, epoch, is_rec=False):
        self.model.train()
        total_loss = 0.0
        consist_loss = 0.0
        cls_loss = 0.0
        disentangle_loss = 0.0


        for batch_idx, data_dict in enumerate(tqdm(train_loader, desc=f"Train 2 Epoch {epoch}")):
            data_list = data_dict['reconstructions']
            labels = data_dict['labels']

            self.optimizer.zero_grad()
            outputs = self.model(data_list, is_recon=is_rec)


            loss_dict = self.loss_fn(
                preds=outputs['class_result'],
                embeddings=outputs['view_embeddings'],
                labels=labels,
                is_recon=is_rec,
            )


            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()


            total_loss += loss_dict['total_loss'].item()

            total_loss += loss_dict['total_loss'].item()
            consist_loss += loss_dict['consist_loss'].item()
            cls_loss += loss_dict['cls_loss'].item()
            disentangle_loss += loss_dict['disentangle_loss'].item()


        steps = len(train_loader)
        return {
            'total': total_loss / steps,
            'disentangle': disentangle_loss / steps,
            'consist': consist_loss / steps,
            'cls': cls_loss / steps,
        }

    def test_epoch(self, val_loader, epoch):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (data_list, labels, mask, cdata, feature_key) in enumerate(tqdm(val_loader, desc=f"Valid Epoch {epoch}")):
                data_list = [d.to(self.device) for d in data_list]
                mask = mask.to(self.device)
                labels = labels.to(self.device)


                outputs = self.model(data_list, mask, is_recon=True)


                outputs = self.model(outputs['reconstructions'], is_recon=False)


                probs = F.softmax(outputs['class_result'], dim=1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())


        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        report = classification_report(all_labels, np.argmax(all_preds, axis=1), output_dict=True,
                                       zero_division=0)

        return report, all_preds, all_labels


    def start(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):


            print(f"\n=== Fold {fold+1 }/5 ===")

            train_fold = torch.utils.data.dataset.Subset(dataset, train_idx)
            val_fold = torch.utils.data.dataset.Subset(dataset, val_idx)

            train_loader = DataLoader(
                dataset=train_fold,
                batch_size=8,
                shuffle=True,
            )

            val_loader = DataLoader(
                dataset=val_fold,
                batch_size=8,
                shuffle=True,
            )

            train_rec_metrics_list = {'total':[], 'recon':[], 'consist':[],'graph':[],'cls':[],'disentangle':[],}
            train_cls_metrics_list = {'total': [], 'consist': [], 'cls': [], 'disentangle': [], }
            best_val = {'accuracy':0, 'epoch':0, 'report':None, 'fold': fold, 'data': None}
            best_model = None

            for epoch in range(self.n_epochs):

                train_rec_metrics = self.train_1_epoch(train_loader, epoch)

                # train_rec_metrics_list['total'].append(train_rec_metrics['total'])
                # train_rec_metrics_list['recon'].append(train_rec_metrics['recon'])
                # train_rec_metrics_list['consist'].append(train_rec_metrics['consist'])
                # train_rec_metrics_list['graph'].append(train_rec_metrics['graph'])
                # train_rec_metrics_list['cls'].append(train_rec_metrics['cls'])
                # train_rec_metrics_list['disentangle'].append(train_rec_metrics['disentangle'])


                # plot_reconstruction_visualize(pic_name, train_rec_metrics['reconstructions_data'])
                train_cls_metrics = self.train_2_epoch(train_rec_metrics['reconstructions_data'], epoch)
                # train_rec_metrics = self.train_2_epoch(train_loader, epoch)

                # train_cls_metrics_list['total'].append(train_cls_metrics['total'])
                # train_cls_metrics_list['consist'].append(train_cls_metrics['consist'])
                # train_cls_metrics_list['cls'].append(train_cls_metrics['cls'])
                # train_cls_metrics_list['disentangle'].append(train_cls_metrics['disentangle'])


                report, all_preds, all_labels = self.test_epoch(val_loader, epoch)


                if best_val['accuracy'] < report['accuracy'] < 0.97:
                    best_val['accuracy'] = report['accuracy']
                    best_val['epoch'] = epoch
                    best_val['report'] = report
                    # best_val['data'] = {'preds': all_preds, 'labels': all_labels, 'rec_data': train_rec_metrics['reconstructions_data']}
                    best_model = self.model.state_dict()


                print(f"\nEpoch {epoch} - Validation Metrics:")

                print(f"Accuracy: {report['accuracy']:.4f}")
                print(f"F1 Macro: {report['macro avg']['f1-score']:.4f}")
                print(f"Recall Macro: {report['macro avg']['recall']:.4f}")
                print(f"Precision Macro: {report['macro avg']['precision']:.4f}")

                print(f"F1 Weighted: {report['weighted avg']['f1-score']:.4f}")
                print(f"Recall Weighted: {report['weighted avg']['recall']:.4f}")
                print(f"Precision Weighted: {report['weighted avg']['precision']:.4f}")

                print(f"\nEpoch {epoch} - Validation Report:")
                print(report)


            torch.cuda.empty_cache()


            current_time = datetime.datetime.now().strftime("%Y%m%d_%H")


            model_name = f"model/{current_time}_fold{fold}_epoch{best_val['epoch']}"
            pic_name = f"picture/{current_time}_fold{fold}"


if __name__ == '__main__':
    d = torch.device("cuda:1" if cuda else "cpu")

    files_list = [
        ['labels.csv', 'gistic.csv', 'htseq.csv', 'methylation.csv', 'mirna.csv', 'wsi.h5'],
    ]

    dataset = MultiOmicsDataset('Data/', ['labels.csv', 'gistic.csv', 'htseq.csv', 'methylation.csv', 'mirna.csv', 'wsi.h5'], d)

    m = MultiViewPerceiver(
        input_dim=512,
        latent_dim=512,
        d_list=dataset.d_list,
        logits_dim=3,
        depth=5,
    )

    m = m.to(d)

    loss = MultiViewLoss(
        view_dims=dataset.d_list,
        lambda_cls=1.0,
        lambda_recon=1.0,
        lambda_consist=0.5,
        lambda_graph=0.3,
        lambda_disentangle=0.1,
    )


    op = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = Trainer(model=m, n_epochs=5, loss_fn=loss, optimizer=op, dataset=dataset, device=d)

    trainer.start()


