# Global imports
import torch

# pytorch_metric_learning package imports
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning import reducers
from pytorch_metric_learning.reducers.base_reducer import BaseReducer


# Clone of MeanReducer
class SafeMeanReducer(BaseReducer):
    def forward(self, loss_dict, embeddings, labels):
        try:
            n = loss_dict['loss']['indices'][0].size(0)
        except TypeError:
            print('WARNING: no losses')
            # Trick to return a tensor with gradient 0 so distributed doesn't break
            return embeddings[0, 0] * 0.0
        # To compute mean: divide by total before taking sum to avoid overflow (esp. fp16)
        # - may instead result in underflow, but that is less problematic
        loss = (loss_dict['loss']['losses'] / n).sum()
        # Make sure loss is not nan
        if loss.isnan() or loss.isinf():
            print('WARNING: SafeNTXentLoss is nan or inf')
            return torch.tensor(0.0).to(loss.device)
        else:
            return loss


# Clone of NTXentLoss
class SafeNTXentLoss(GenericPairLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.reducer = SafeMeanReducer()

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return CosineSimilarity()
