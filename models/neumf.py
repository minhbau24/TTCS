import torch
import torch.nn as nn

# Lớp mô hình NeuMF
class NeuMF(nn.Module):
    def __init__(self, num_customers, num_products, num_sellers, cat_vocab_sizes, num_features=2, embed_dim=32, mlp_layers=[128, 64, 32]):
        super(NeuMF, self).__init__()
        self.customer_embed_gmf = nn.Embedding(num_customers, embed_dim)
        self.product_embed_gmf = nn.Embedding(num_products, embed_dim)
        
        self.customer_embed_mlp = nn.Embedding(num_customers, embed_dim)
        self.product_embed_mlp = nn.Embedding(num_products, embed_dim)
        self.seller_embed_mlp = nn.Embedding(num_sellers, embed_dim)

        self.cat_level_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim, padding_idx=0) for vocab_size in cat_vocab_sizes
        ]) if cat_vocab_sizes else nn.ModuleList()

        mlp_input_dim = embed_dim * (3 + len(cat_vocab_sizes)) + num_features
        mlp_layers_list = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            mlp_layers_list.append(nn.Linear(in_dim, out_dim))
            mlp_layers_list.append(nn.ReLU())
            mlp_layers_list.append(nn.Dropout(0.3))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers_list)
        
        self.final_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, customer, product, seller, features, cat_levels, custom_gmf_embed=None, custom_mlp_embed=None):
        customer_gmf = custom_gmf_embed if custom_gmf_embed is not None else self.customer_embed_gmf(customer)
        product_gmf = self.product_embed_gmf(product)
        gmf_output = customer_gmf * product_gmf

        customer_mlp = custom_mlp_embed if custom_mlp_embed is not None else self.customer_embed_mlp(customer)
        product_mlp = self.product_embed_mlp(product)
        seller_mlp = self.seller_embed_mlp(seller)
        
        if self.cat_level_embeds:
            cat_embeds = [embed(cat_levels[:, i]) for i, embed in enumerate(self.cat_level_embeds)]
            cat_embeds = torch.cat(cat_embeds, dim=1)
        else:
            cat_embeds = torch.zeros(customer.shape[0], 0, device=customer.device)

        mlp_input = torch.cat([customer_mlp, product_mlp, seller_mlp, cat_embeds, features], dim=1)
        mlp_output = self.mlp(mlp_input)

        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.final_layer(combined)
        return self.sigmoid(output).squeeze()