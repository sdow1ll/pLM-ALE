from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerFast
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc

class ProteinDataset:
    def __init__(self, sequences, tokenizer):
            self.sequences = sequences
            self.tokenizer = tokenizer

    def __len__(self):
            return len(self.sequences)

    def __getitem__(self, idx):
            item = self.sequences[idx]
            return self.tokenizer(item, truncation=True, padding='max_length', max_length=512, return_tensors="pt")

class ProtGPT()

class ProteinLanguageModel:
    """Generalized protein language model class."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        model_class: type = AutoModelForMaskedLM,
        tokenizer_class: type = AutoTokenizer,
        half_precision: bool = False,
        batch_size: int = 2,
    ):
        """Initialize the protein language model.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            The model id or path to load the pretrained model from.
        tokenizer_name_or_path : str, optional
            The tokenizer id or path to load the tokenizer from.
        model_class : type, optional
            The model class to use, by default AutoModelForMaskedLM.
        tokenizer_class : type, optional
            The tokenizer class to use, by default AutoTokenizer.
        half_precision : bool, optional
            Use the model in half precision, by default False.
        batch_size : int, optional
            The batch size to use for the model, by default 1.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path or pretrained_model_name_or_path
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.half_precision = half_precision
        self.batch_size = batch_size

        self._tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_name_or_path)

    @property
    def name(self) -> str:
        """Get the model name."""
        return self.pretrained_model_name_or_path

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Get the tokenizer."""
        return self._tokenizer


    def _prepare_data(self, sequences: list[str]) -> DataLoader:
        """Prepare the data for the model."""
        # Create the dataset
        dataset = ProteinDataset(sequences, self.tokenizer)

        # Create the data collator
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True)

        # Create the data loader
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collator)

        return dataloader

    def _prepare_model(self) -> PreTrainedModel:
        """Prepare the model for inference."""
        # Load model
        model = self.model_class.from_pretrained(self.pretrained_model_name_or_path)

        # Convert the model to half precision if needed
        if self.half_precision:
            model.half()

        # Load the model onto the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)

        return model

    def infer(self, sequences: list[str]) -> list[dict]:
        """Run the model on the sequences.

        Parameters
        ----------
        sequences : list[str]
            The sequences to get the model output for.

        Returns
        -------
        list[dict]
            The model output for each sequence.
        """
        # Prepare the model
        model = self._prepare_model()

        # Prepare the data
        dataloader = self._prepare_data(sequences)

        model_outputs = []
        for batch in tqdm(dataloader):
            # Move the batch to the device
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # Get the model outputs with a forward pass
            outputs = model(**inputs, output_hidden_states=True)

            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Extract embeddings and logits
            sequence_embeddings = last_hidden_state.mean(dim=1)
            logits = outputs.logits.cpu().detach().numpy()

            # Store the outputs
            model_outputs.append({
                "logits": logits,
                "sequence_embeddings": sequence_embeddings.cpu().detach().numpy(),
                "residue_embeddings": last_hidden_state.cpu().detach().numpy()
            })

        # Clean up the model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Return the outputs
        return model_outputs
