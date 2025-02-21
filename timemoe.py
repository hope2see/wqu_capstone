import os
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from abstractmodel import AbstractModel
from mem_util import MemUtil

from transformers import AutoModelForCausalLM

_mem_util = MemUtil(rss_mem=False, python_mem=False)

def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


class TimeMoE(AbstractModel):
    def __init__(self, configs):
        super().__init__(configs, "TimeMoE")

        # from run_eval.py of Time-MoE ------ 
        model_path = 'Maple728/TimeMoE-50M'        
        # master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
        # master_port = os.getenv('MASTER_PORT', 9899)
        # world_size = int(os.getenv('WORLD_SIZE') or 1)
        # rank = int(os.getenv('RANK') or 0)
        # local_rank = int(os.getenv('LOCAL_RANK') or 0)
        # if torch.cuda.is_available():
        #     try:
        #         setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
        #         device = f"cuda:{local_rank}"
        #         is_dist = True
        #     except Exception as e:
        #         print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
        #         device = 'cpu'
        #         is_dist = False
        # else:
        #     device = 'cpu'
        #     is_dist = False
            
        # from NeuralNetModel 
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


    # from NeuralNetModel 
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
        # TimeMoE expects --
        # 'inputs': np.array(window_seq[: self.context_length], dtype=np.float32),
        # 'labels': np.array(window_seq[-self.prediction_length:], dtype=np.float32),

        # reshape batch_x to (feature_dim, seq_len)
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
