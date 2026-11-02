"""Runnable DocVLM topology. Paper-scale backbones are configurable external integrations."""
from torch import nn
from src.models.vision_encoder import PatchVisionEncoder
from src.models.document_encoder import OCRLayoutEncoder
from src.models.heads import VLMProjection
from src.models.text_encoder import TinyCausalDecoder
from src.models.adaptive_query_compression import QuestionAwareQuerySelector
class DocVLM(nn.Module):
    def __init__(self,c):
        super().__init__();h,v=c['ocr_hidden_size'],c['vlm_hidden_size'];self.vision_encoder=PatchVisionEncoder(v,c['patch_size']);self.ocr_encoder=OCRLayoutEncoder(c['vocab_size'],h,c['num_queries'],c['ocr_layers'],c['num_heads'],c['dropout']);self.projection=VLMProjection(h,v);self.decoder=TinyCausalDecoder(c['vocab_size'],v,c['decoder_layers'],c['num_heads'],c['dropout']);self.compression_type=c.get('compression_type','fixed');self.selector=QuestionAwareQuerySelector(h,c['num_queries'],c.get('adaptive_k',c['num_queries'])) if self.compression_type=='adaptive' else None
        if c.get('freeze_vlm',False):
            for module in (self.vision_encoder,self.decoder):
                for p in module.parameters():p.requires_grad=False
    def forward(self,pixel_values,ocr_ids,ocr_boxes,ocr_mask,question_ids,answer_ids):
        visual=self.vision_encoder(pixel_values); candidates=self.ocr_encoder(ocr_ids,ocr_boxes,ocr_mask,question_ids); scores=indices=None
        if self.selector is not None: candidates,scores,indices=self.selector(candidates,self.ocr_encoder.char(question_ids))
        compressed=self.projection(candidates)
        return {'logits':self.decoder(__import__('torch').cat([visual,compressed],1),answer_ids),'ocr_features':compressed,'visual_features':visual,'query_scores':scores,'selected_query_indices':indices}
