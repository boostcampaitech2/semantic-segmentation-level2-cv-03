"""
Locate the codes below into [your conda env]/mmseg/models/segmentors/encoder_decoder.py
"""


import os
import torch
import pickle
import numpy as np

# class EncoderDecoder(BaseSegmentor):
#     """Encoder Decoder segmentors.

#     EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
#     Note that auxiliary_head is only used for deep supervision during training,
#     which could be dumped during inference.
#     """
    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            # print(imgs[i].shape)
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)

        is_conf = False
        if is_conf:
            print('\n'*5)
            print(f'adding confidences {len(self.logits4ref)}')
            print('\n'*5)
            # self.logits4ref.append(torch.softmax(seg_logit.squeeze(), dim=0).cpu().numpy())
            # 1 * 11 * 512 512
            self.logits4ref.append(torch.softmax(seg_logit.squeeze(), dim=0).cpu().numpy().astype(np.float16))
        
        ref_dir = '/opt/ml/segmentation/mmsegmentation/submission/refConfidence'
        if len(self.logits4ref) == 819:
            pkl_name = 'UperSwinB_final_fold1.pkl'
            # pkl_name = 'BEiT_JSW_pse_1029LB734.pkl'
            # pkl_name = 'ERROR.pkl'
            with open(os.path.join(ref_dir, pkl_name), 'wb') as f:
                pickle.dump(self.logits4ref, f)
                print(f'{pkl_name} dumped!')

        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
