import numpy as np

from transformers import T5Tokenizer
from transformers.processing_utils import ProcessorMixin


class UioProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "UioImageProcessor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, image_processor):
        tokenizer = T5Tokenizer.from_pretrained(
            "t5-base", model_max_length=256, extra_ids=1100
        )

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.max_input_len = 256
        self.pad_to_max = True

    def __call__(self, images, text, masks=None):

        batch = {}

        image_encoder_inputs, image_input_masks = self.image_processor(images, masks)
        batch["image_encoder_inputs"] = image_encoder_inputs
        batch["image_input_masks"] = image_input_masks

        input_tokens = np.array(
            self.tokenizer(
                text,
                max_length=self.max_input_len,
                truncation=True,
                padding="max_length" if self.pad_to_max else "longest",
            )["input_ids"],
            dtype=np.int32,
        )
        batch["input_tokens"] = input_tokens
        return batch
