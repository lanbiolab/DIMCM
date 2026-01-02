
import torch
from torch import nn
import torch.nn.functional as F

from RecFormer.Perceiver.perceiver_io import PerceiverIO
from RecFormer.RFPIOProject.utils import exists


class MultiViewPerceiver(nn.Module):
    def __init__(self, input_dim, d_list, cross_view_heads=4, logits_dim=3, depth=4, heads=8, latent_dim=512, dropout=0.1):

        super().__init__()
        self.perceiver = PerceiverIO(
            depth=depth,
            dim=input_dim,
            queries_dim=input_dim,
            latent_dim=latent_dim,
            num_latents=256,
            cross_heads=heads,
            latent_heads=heads,
            decoder_ff=True
        )


        self.embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, input_dim),
                nn.LayerNorm(input_dim),
            ) for d in d_list
        ])


        self.reconstruction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, d)
            ) for d in d_list
        ])


        self.cross_view_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim,
                                  num_heads=cross_view_heads)
            for _ in range(depth // 2)
        ])


        self.view_queries = nn.Parameter(
            torch.randn(len(d_list), input_dim))  # [V, D]


        self.mask_token = nn.Parameter(torch.randn(1, input_dim))


        self.view_weight_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in d_list
        ])


        self.to_logits = nn.Linear(latent_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, x_list, mask=None, is_recon=True):

        batch_size = x_list[0].shape[0]
        num_views = len(x_list)

        embeddings = []
        if mask is not None:
            for v in range(num_views):

                masked = torch.where(
                    mask[:, v].unsqueeze(-1).bool(),
                    self.embedding[v](x_list[v]),
                    self.mask_token.expand(x_list[v].shape[0], -1)
                )
                embeddings.append(masked)
        else:
            for v in range(num_views):
                embeddings.append(self.embedding[v](x_list[v]))


        x = torch.stack(embeddings, dim=1)
        # B, V, D = x.shape


        for i, attn_layer in enumerate(self.cross_view_attn):

            x_attn = x.permute(1, 0, 2)
            attn_out, _ = attn_layer(x_attn, x_attn, x_attn)
            x = x + 0.5 * attn_out.permute(1, 0, 2)


        latent = self.perceiver(
            x,
            mask=mask.unsqueeze(-1) if mask is not None else None,
            queries=None
        )

        if is_recon:

            active_queries = torch.index_select(
                self.view_queries,
                0,
                torch.arange(num_views, device=x.device)
            ).unsqueeze(0).expand(batch_size, -1, -1)  # [B, V, D]


            view_weights = []
            for v in range(num_views):
                weight = self.view_weight_network[v](embeddings[v])
                view_weights.append(weight)

            view_weights = torch.cat(view_weights, dim=1)  # [B, V]


            weighted_queries = active_queries * view_weights.unsqueeze(-1)


            reconstructions = self.perceiver(
                x,
                mask=mask.unsqueeze(-1) if mask is not None else None,
                queries=weighted_queries
            )  # [B, V, D]


            raw_recons = [self.reconstruction[v](reconstructions[:, v]) for v in range(num_views)]


            x_bar = []
            for v in range(num_views):

                original_data = x_list[v].to(raw_recons[v].device)

                missing_mask = ~mask[:, v].unsqueeze(-1).bool()

                x_bar_v = torch.where(missing_mask, raw_recons[v], original_data)
                x_bar.append(x_bar_v)

            result = self.to_logits(latent.mean(dim=1))

            return {
                "class_result": result,
                "reconstructions": x_bar,
                "latent": latent,
                "reconstructions_embeddings": reconstructions,
                "view_embeddings": x,
                "view_weights": view_weights,
            }
        else:
            result = self.to_logits(latent.mean(dim=1))

            return {
                "class_result": result,
                "latent": latent,
                "view_embeddings": x,
            }


class MultiViewLoss(nn.Module):
    def __init__(self,
                 view_dims,
                 temperature=0.1,
                 lambda_cls=1.0,
                 lambda_recon=1.0,
                 lambda_consist=0.5,
                 lambda_graph=0.3,
                 lambda_disentangle=0.1,
                 epsilon=1e-8):
        super().__init__()
        self.temp = temperature
        self.epsilon = epsilon


        self.lambda_cls = nn.Parameter(torch.tensor(lambda_cls))
        self.lambda_disentangle = nn.Parameter(torch.tensor(lambda_disentangle))
        self.lambda_recon = nn.Parameter(torch.tensor(lambda_recon))
        self.lambda_consist = nn.Parameter(torch.tensor(lambda_consist))
        self.lambda_graph = nn.Parameter(torch.tensor(lambda_graph))
        self.temperature = nn.Parameter(torch.tensor(temperature))

        self.view_weights = nn.Parameter(torch.ones(len(view_dims)))


        # self.graph_sim = nn.CosineSimilarity(dim=2)


        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def safe_norm(self, x):

        return x / (torch.norm(x, dim=-1, keepdim=True) + self.epsilon)

    @staticmethod
    def disentangle_loss(embeddings):
        # embeddings: [B, V, D]
        cov_matrix = torch.cov(embeddings.reshape(-1, embeddings.size(-1)).t())
        diag = torch.diag(cov_matrix).mean()
        off_diag = (cov_matrix.sum() - diag) / (cov_matrix.size(0) ** 2 - cov_matrix.size(0))
        return torch.abs(off_diag)

    def cls_loss(self, preds, labels):

        loss = torch.mean(self.criterion(preds, labels.to(torch.int64)))

        return loss

    def weighted_reconstruction_loss(self, reconstruction, row_data, masks):
        recon_loss = 0.0
        valid_pixels = 0

        for v, (pred, target) in enumerate(zip(reconstruction, row_data)):

            missing_mask = ~masks[:, v].unsqueeze(-1).bool()  # [B, 1]
            loss = (missing_mask * (target - pred) ** 2).sum()
            recon_loss += self.view_weights[v] * loss
            valid_pixels += missing_mask.sum()

        return recon_loss / (valid_pixels + self.epsilon)

    def view_consistency_loss(self, embeddings):
        # embeddings: [B, V, D]
        B, V, D = embeddings.shape


        norm_emb = F.normalize(embeddings, dim=-1)  # [B, V, D]
        sim_matrix = torch.einsum('bvd,bwd->bvw', norm_emb, norm_emb)  # [B, V, V]


        targets = torch.eye(V, device=embeddings.device).unsqueeze(0)  # [1, V, V]
        targets = targets.expand(B, -1, -1)  # [B, V, V]


        loss = F.binary_cross_entropy_with_logits(
            sim_matrix / self.temperature.clamp(min=0.01),
            targets,
            reduction='none'
        ).mean(dim=(1, 2)).mean()

        return loss

    @staticmethod
    def graph_regularization_loss(reconstruction_data_adj_matrix, adj_matrix):

        return F.mse_loss(reconstruction_data_adj_matrix, adj_matrix)

    def forward(self,
                preds,
                labels,
                row_data=None,
                reconstructions_data=None,
                row_adj_matrix=None,
                reconstructions_adj_matrix=None,
                masks=None,
                embeddings=None,
                is_recon=True,
                ):

        cls_loss = self.cls_loss(preds, labels)


        recon_loss = 0
        if reconstructions_data is not None:
            recon_loss = self.weighted_reconstruction_loss(reconstructions_data, row_data, masks)


        consist_loss = 0
        if embeddings is not None:
            consist_loss = self.view_consistency_loss(embeddings)


        graph_loss = 0.0
        if row_adj_matrix is not None:
            # combined_embed = torch.stack(embeddings).mean(dim=0)
            graph_loss = self.graph_regularization_loss(reconstructions_adj_matrix, row_adj_matrix)


        disentangle_loss = 0
        if embeddings is not None:
            disentangle_loss = self.disentangle_loss(embeddings)



        total_loss = (
                torch.sigmoid(self.lambda_cls) * cls_loss +
                0 * torch.sigmoid(self.lambda_recon) * recon_loss +
                torch.sigmoid(self.lambda_consist) * consist_loss +
                torch.sigmoid(self.lambda_graph) * graph_loss +
                torch.sigmoid(self.lambda_disentangle) * disentangle_loss
        )
        if is_recon:
            return {
                "total_loss": total_loss,
                "cls_loss": cls_loss.detach(),
                "recon_loss": recon_loss.detach(),
                "consist_loss": consist_loss.detach(),
                "graph_loss": graph_loss.detach(),
                "disentangle_loss": disentangle_loss.detach(),
                "view_weights": self.view_weights.detach().cpu(),
            }
        else:
            return {
                "total_loss": total_loss,
                "cls_loss": cls_loss.detach(),
                "consist_loss": consist_loss.detach(),
                "disentangle_loss": disentangle_loss.detach(),
                "view_weights": self.view_weights.detach().cpu(),
            }

    def feature_importance(self, model, data_list, labels, mask):
        model.eval()
        data_list = [d.requires_grad_() for d in data_list]
        outputs = model(data_list, mask)
        loss = self.cls_loss(outputs['class_result'], labels)
        loss.backward()

        importance = [torch.norm(d.grad, dim=1).mean().item() for d in data_list]
        return importance
