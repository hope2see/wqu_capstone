import os
import torch
from transformers import AutoModelForCausalLM
from tabe.models.abstractmodel import AbstractModel

from tabe.utils.mem_util import MemUtil
_mem_util = MemUtil(rss_mem=False, python_mem=False)


class TimeMoE(AbstractModel):
    def __init__(self, configs):
        super().__init__(configs, "TimeMoE")

        model_path = 'Maple728/TimeMoE-50M'        
            
        device = self._acquire_device()

        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        # logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        # self.prediction_length = configs.pred_len
        self.model.eval()


    def _acquire_device(self):
        if self.configs.use_gpu and self.configs.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.configs.gpu) if not self.configs.use_multi_gpu else self.configs.devices
            device = torch.device('cuda:{}'.format(self.configs.gpu))
            print('Use GPU: cuda:{}'.format(self.configs.gpu))
        # NOTE 
        # It seems we can't use 'mps' for TimeMoE model, 
        # because TimeMoE model use float64 dtype and mps doesn't support float64.
        #  "TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead."
        #
        # elif self.configs.use_gpu and self.configs.gpu_type == 'mps':
        #     device = torch.device('mps')
        #     print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device


    # NOTE
    # Pretrained model. So, it not necessary to train the model.
    # However, it would be better to fine-tune the model with the dataset.
    # TODO : Fine-tune
    def train(self):
        pass

    def test(self):
        raise NotImplementedError

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # Given: batch_x.shape = (batch_len=1, seq_len, feature_dim)
        # TimeMoE expect:
        #   'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
        #   'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),
        # So, reshape batch_x to (feature_dim, seq_len)
        batch_x = batch_x[0].T
        
        outputs = self.model.generate(
            inputs=batch_x.to(self.device).to(self.model.dtype),
            max_new_tokens=1, # prediction_length
        )
        y_hat = outputs[-1, -1].item()

        # calculate the actuall loss of next timestep
        y = batch_y[0, -1:, -1] 
        loss = criterion(torch.tensor(y_hat), y).item()

        if training: # TODO 
            pass

        return y_hat, loss
