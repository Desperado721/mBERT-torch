import torch.nn as nn
class BERTClassifier(nn.Module):
    def __init__(self, bert_encoder, args):
        super(BERTClassifier, self).__init__()
        self.encoder = bert_encoder
        self.hidden_dim = args.hidden_dim
        self.dropout = args.p_dropout
        self.fc_dim = args.fc_dim
        self.out_dim = args.out_dim
        self.activation = nn.Tanh()
        # self.softmax = nn.softmax()
        self.mlp = nn.Sequential(nn.Dropout(p=self.dropout),
                    nn.Linear(self.hidden_dim,self.out_dim))
                    # self.activation,
                    # nn.Dropout(p=self.dropout),
                    # self.activation,
                    # nn.Linear(self.fc_dim,self.out_dim))
        # self.softmax = nn.softmax()

    def forward(self, input_ids=None, attention_mask=None):
        encoder_output = self.encoder(input_ids=input_ids,attention_mask=attention_mask) #[num]
        last_hidden = encoder_output[0]
        attention = encoder_output[2]
        # [CLS]
        logits = self.mlp(encoder_output[1])
        return logits, attention, last_hidden

class ModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for
                     instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args.batch_size),
                      'Building a local approximation...'):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = max([len(_l) for _l in batch_ids])
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(tokens_tensor)
            results += logits.detach().cpu().numpy().tolist()
        return np.array(results)

class BertModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(BertModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for
                     instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args.batch_size),
                      'Building a local approximation...'):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = min(max([len(_l) for _l in batch_ids]), 512)
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(tokens_tensor.long(),
                                attention_mask=tokens_tensor.long() > 0)
            results += logits[0].detach().cpu().numpy().tolist()
        return np.array(results)