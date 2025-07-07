"""Evolve sequences with language models.

Inspired from:
https://github.com/brianhie/efficient-evolution
"""

from __future__ import annotations

import gc
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from peft import PeftModel
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmForMaskedLM
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import EsmTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from autoamp.finetune import Sequence

def avg_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Check that the batch size of embeddings and attention_mask match
    if embeddings.size(0) != attention_mask.size(0):
        raise ValueError(f"Batch size of embeddings ({embeddings.size()}) does not match batch size of attention mask ({attention_mask.size()})")

    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)
    #print('seq_lengths size: 2',seq_lengths.size())
    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[torch.arange(attention_mask.size(0)), seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, 1)
    pool_mask = attention_mask.unsqueeze(-1)  # Shape becomes (B, SeqLen, 1)

    # Now expand the pool_mask to match the embeddings' dimensions (B, SeqLen, HiddenDim)
    pool_mask = pool_mask.expand_as(embeddings)

    # Sum the embeddings over the sequence length (use the mask to avoid padding tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, dim=1)

    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(dim=1), min=1e-9)

    # Compute mean pooled embeddings for each sequence
    return sum_embeds / sum_mask


def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)

    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[:, seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape)

    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, 1)

    # Avoid division by zero for zero length sequences by clamping
    sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)

    # Compute mean pooled embeddings for each sequence
    return sum_embeds / sum_mask


from peft import PeftModel
from transformers import EsmForMaskedLM, EsmTokenizer

class LoRAEsm2ProteinLanguageModel:
    """LoRA fine-tuned protein language model for ESM2."""

    def __init__(
        self,
        base_model_name: str = 'facebook/esm2_t33_650M_UR50D',
        checkpoint_path: str = "path/to/your/lora_checkpoint",
        tokenizer_name: str = 'facebook/esm2_t33_650M_UR50D',
        half_precision: bool = False,
        batch_size: int = 1,
    ):
        """Initialize the LoRA fine-tuned model."""
        self.base_model_name = base_model_name
        self.checkpoint_path = checkpoint_path
        self.tokenizer_name = tokenizer_name
        self.half_precision = half_precision
        self.batch_size = batch_size
    
        # Load appropriate tokenizer based on model type
        if "progen" in self.base_model_name.lower():
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                trust_remote_code=True
            )
            # Handle padding token for ProGen models
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self._tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)

    @property
    def tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    def _prepare_model(self):
        """Load the base model and apply LoRA fine-tuned weights."""
        if "progen" in self.base_model_name.lower():
            from transformers import AutoModelForCausalLM
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, 
                trust_remote_code=True
            )
        else:
            base_model = EsmForMaskedLM.from_pretrained(self.base_model_name)
    
        # Load LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
    
        # Use half precision if required
        if self.half_precision:
            peft_model.half()
    
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        peft_model.to(self.device)
    
        return peft_model

    def infer(self, sequence: str) -> ProteinLanguageModelOutput:
        """Run inference and return ProteinLanguageModelOutput."""
        # Load model
        model = self._prepare_model()

        # Tokenize input
        inputs = self.tokenizer(sequence, return_tensors="pt").to(model.device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract logits
        logits = outputs.logits.cpu().detach().numpy()

        # Extract last hidden state (sequence embeddings)
        last_hidden_state = outputs.hidden_states[-1]
        sequence_embeddings = last_hidden_state.mean(dim=1).cpu().detach().numpy()  # Average pooling

        # Residue embeddings (remove start/end tokens)
        seq_len = inputs["attention_mask"].sum().item() - 2  # Ignore start/end tokens
        residue_embeddings = last_hidden_state[:, 1:seq_len+1, :].cpu().detach().numpy()

        # Return structured output
        return ProteinLanguageModelOutput(
            logits=logits[0, 1:seq_len+1, :],  # Remove special tokens
            sequence_embeddings=sequence_embeddings[0],
            residue_embeddings=residue_embeddings[0]
        )

    def __call__(self, input_ids):
        """
        Make the model callable directly, allowing it to be used with HeatmapGenerator.
        Properly handles BatchEncoding objects from tokenizers.
        
        Args:
            input_ids: Tensor of input token IDs, dict with inputs, or BatchEncoding
            
        Returns:
            Model outputs with logits
        """
        # Load the model (this is similar to what you do in infer method)
        model = self._prepare_model()
        
        # Move input_ids to the correct device
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Handle different input types
        inputs = {}
        
        # Check if it's a BatchEncoding object (from tokenizer)
        if hasattr(input_ids, 'input_ids'):
            for key, value in input_ids.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                else:
                    inputs[key] = value
                    
        # Check if it's a dictionary
        elif isinstance(input_ids, dict):
            for key, value in input_ids.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                else:
                    inputs[key] = value
                    
        # Assume it's a tensor otherwise
        else:
            inputs["input_ids"] = input_ids.to(device)
            # Don't try to create attention_mask here - this is what caused the error
        
        # Run inference with the model
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        return outputs


class Esm2ProteinLangaugeModel:
    """Protein language model for the ESM-2 model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'facebook/esm2_t6_8M_UR50D',
        tokenizer_name_or_path: str = 'facebook/esm2_t6_8M_UR50D',
        half_precision: bool = False,
        batch_size: int = 1,
    ):
        """Initialize the pLM.

        Parameters
        ----------
        pretrained_model_name_or_path : str, optional
            The model id or path to load the pretrained model from,
            by default 'facebook/esm2_t6_8M_UR50D'
        tokenizer_name_or_path : str, optional
            The tokenizer id or path to load the tokenizer from,
            by default 'facebook/esm2_t6_8M_UR50D'
        half_precision : bool, optional
            Use the model in half precision, by default False
        batch_size : int, optional
            The batch size to use for the model, by default 1
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.half_precision = half_precision
        self.batch_size = batch_size

        self._tokenizer = EsmTokenizer.from_pretrained(tokenizer_name_or_path)

    @property
    def name(self) -> str:
        """Get the model name."""
        return self.pretrained_model_name_or_path

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Get the tokenizer."""
        return self._tokenizer

    def _prepare_data(self, sequences: list[Sequence]) -> DataLoader:
        """Prepare the data for the model."""
        from autoamp.finetune import DataCollator
        from autoamp.finetune import SequenceDataset

        # Extract the string sequences
        seqs = [seq.sequence for seq in sequences]

        # Create the dataset
        dataset = SequenceDataset(seqs)

        # Create the data collator
        collater_fn = DataCollator(
            train_mode=False, tokenizer=self.tokenizer, mlm=True
        )

        # Create the data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collater_fn,
        )

        return dataloader

    def _prepare_model(self) -> PreTrainedModel:
        """Prepare the model for inference."""
        # Load model and tokenizer
        model = EsmForMaskedLM.from_pretrained(
            self.pretrained_model_name_or_path
        )

        # Convert the model to half precision
        if self.half_precision:
            model.half()

        # Load the model onto the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(self.device)

        return model

    def infer(
        self, sequences: list[Sequence]
    ) -> list[ProteinLanguageModelOutput]:
        """Run the model on the sequences.

        Parameters
        ----------
        sequences : list[str]
            The sequences to get the model output for.

        Returns
        -------
        list[ProteinLanguageModelOutput]
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

            # Get the sequence lengths (remove the end token)
            seq_lengths = inputs['attention_mask'].sum(axis=1) - 1

            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Get the sequence embeddings
            sequence_embeds = average_pool(
                last_hidden_state, inputs['attention_mask']
            )

            # Move the outputs to the CPU
            logits = outputs.logits.cpu().detach().numpy()
            sequence_embeds = sequence_embeds.cpu().detach().numpy()
            residue_embeds = last_hidden_state.cpu().detach().numpy()

            # Create the output objects
            for i, seq_len in enumerate(seq_lengths):
                # Remove the start, end and padding tokens
                logit = logits[i, 1:seq_len, :]
                sequence_embed = sequence_embeds[i]
                residue_embed = residue_embeds[i, 1:seq_len, :]

                # Create the output object
                output = ProteinLanguageModelOutput(
                    logits=logit,
                    sequence_embeddings=sequence_embed,
                    residue_embeddings=residue_embed,
                )
                model_outputs.append(output)

        # Clean up the model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Return the outputs
        return model_outputs
        
class Progen2_ProteinLangaugeModel:
    """Protein language model for the progen2 model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'hugohrban/progen2-small',
        tokenizer_name_or_path: str = 'hugohrban/progen2-small',
        half_precision: bool = False,
        batch_size: int = 2,
        pad_token: str = None
    ):
        """Initialize the pLM.

        Parameters
        ----------
        pretrained_model_name_or_path : str, optional
            The model id or path to load the pretrained model from,
        tokenizer_name_or_path : str, optional
            The tokenizer id or path to load the tokenizer from,
        half_precision : bool, optional
            Use the model in half precision, by default False
        batch_size : int, optional
            The batch size to use for the model, by default 1
        """
        from transformers import AutoTokenizer
        
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.half_precision = half_precision
        self.batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        # Handling ProGen2's lack of padding token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            
    @property
    def name(self) -> str:
        """Get the model name."""
        return self.pretrained_model_name_or_path

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Get the tokenizer."""
        return self._tokenizer

    def _prepare_data(self, sequences: list[Sequence]) -> DataLoader:
        """Prepare the data for the model."""
        from autoamp.finetune import DataCollator
        from autoamp.finetune import SequenceDataset

        def is_amino_acid_sequence(seq):
            # Define a set of valid amino acid characters
            amino_acids = set("ARNDCEQGHILKMFPSTWYV")
    
            # Check if all characters in the sequence are valid amino acids
            return all(residue in amino_acids for residue in seq)

        # Extract the string sequences
        seqs = [seq.sequence for seq in sequences if is_amino_acid_sequence(seq.sequence)]
        self.seqs = seqs
        
        # Tokenize the sequences beforehand in order for the datacollator to process correctly
        tokenized_seqs = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # sanity check:
        #print('Tokenized Seqs are: \n', tokenized_seqs["input_ids"])
              
        # Create the dataset
        dataset = SequenceDataset(seqs)
        #print('The dataset is: \n', dataset)
        # Create the data collator
        collater_fn = DataCollator(
            tokenizer=self.tokenizer, mlm=False
        )

        # Create the data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collater_fn,
        )

        return dataloader

    def _prepare_model(self) -> PreTrainedModel:
        """Prepare the model for inference."""
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_name_or_path, trust_remote_code=True
        )

        # Convert the model to half precision
        if self.half_precision:
            model.half()

        # Load the model onto the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        model.to(self.device)

        return model
    
    def infer(
        self, sequences: list[Sequence]
    ) -> list[ProteinLanguageModelOutput]:
        """Run the model on the sequences.

        Parameters
        ----------
        sequences : list[str]
            The sequences to get the model output for.

        Returns
        -------
        list[ProteinLanguageModelOutput]
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
            #print('The inputs are: \n',inputs, '\n')

            #tokenized_seqs = self.tokenizer(
            #self.seqs,
            #return_tensors="pt",
            #padding=True,
            #truncation=True
            #)

            #attention_mask = tokenized_seqs['attention_mask'].to(self.device)
            #print('attention mask is: \n', attention_mask)
            #print('attention mask size is: \n', attention_mask.size())

            attention_mask = inputs['attention_mask']
            
            # Get the model outputs with a forward pass
            outputs = model(**inputs, output_hidden_states=True)
            #print('The output logits shape is: \n', outputs.logits.shape)
            
            # Get the sequence lengths (remove the end token)
            seq_lengths = attention_mask.sum(axis=1)
            #print('The seq_lengths are: \n', seq_lengths)

            #seq_lengths.reshape(1,-1) # reshape seq length tensor so that batch size is compatible
            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            #print('The last hidden state shape is: \n', last_hidden_state.shape)
            
            # Get the sequence embeddings
            sequence_embeds = avg_pool(
                last_hidden_state, attention_mask # mH is included for batch compatibility
            )

            # Move the outputs to the CPU
            logits = outputs.logits.cpu().detach().numpy()
            sequence_embeds = sequence_embeds.cpu().detach().numpy()
            residue_embeds = last_hidden_state.cpu().detach().numpy()

            # Create the output objects
            for i, seq_len in enumerate(seq_lengths):
                # Remove the start, end and padding tokens
                logit = logits[i, :seq_len, :]
                sequence_embed = sequence_embeds[i]
                residue_embed = residue_embeds[i, :seq_len, :]

                # Create the output object
                output = ProteinLanguageModelOutput(
                    logits=logit,
                    sequence_embeddings=sequence_embed,
                    residue_embeddings=residue_embed,
                )
                model_outputs.append(output)

        # Clean up the model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Return the outputs
        return model_outputs         

@dataclass
class ProteinLanguageModelOutput:
    """Protein language model output."""

    logits: np.ndarray = field(
        metadata={
            'description': 'The logits of the sequence '
            '(shape: [sequence_length, vocab_size]).'
        }
    )
    sequence_embeddings: np.ndarray = field(
        metadata={
            'description': 'The sequence embeddings '
            '(shape: [embedding_size]).'
        }
    )
    residue_embeddings: np.ndarray = field(
        metadata={
            'description': 'The residue embeddings '
            '(shape: [sequence_length, vocab_size]).'
        }
    )


@dataclass
class Mutation:
    """A mutation in a sequence."""

    position: int = field(metadata={'description': 'The mutation position.'})
    wildtype: str = field(metadata={'description': 'The wildtype amino acid.'})
    mutant: str = field(metadata={'description': 'The mutant amino acid.'})

    def __hash__(self) -> int:
        """Hash the mutation to use it as a key in a dictionary."""
        return hash((self.position, self.wildtype, self.mutant))


@dataclass
class MutatedSequence:
    """A collection of mutated sequences."""

    sequence: Sequence = field(metadata={'description': 'The sequence.'})
    mutation: Mutation = field(
        metadata={'description': 'The mutation in the sequence.'},
    )
    plm_outputs: dict[str, ProteinLanguageModelOutput] = field(
        default_factory=dict,
        metadata={
            'description': 'A dictionary mapping model names to outputs.'
        },
    )

    @property
    def mutated_sequence(self) -> Sequence:
        """Get the mutated sequence."""
        # Copy the wildtype sequence to a list to insert the mutation
        sequence = list(self.sequence.sequence)
        sequence[self.mutation.position] = self.mutation.mutant
        # Create a new sequence with the mutation
        return Sequence(sequence=''.join(sequence), tag=self.sequence.tag)

'''
def mutate_sequence(  
    sequence: Sequence,
    model: LoRAEsm2ProteinLanguageModel,  # Now supports LoRA models
    model_output: ProteinLanguageModelOutput,
    exclude: set[str],
    alpha: float = 0.5,
    skip_first_position: bool = True,
) -> list[MutatedSequence]:
    """Mutate a sequence using logits from a protein language model.

    Parameters
    ----------
    sequence : Sequence
        The sequence to mutate.
    model : LoRAEsm2ProteinLanguageModel
        The fine-tuned LoRA model.
    model_output : ProteinLanguageModelOutput
        The model output with logits.
    exclude : set[str]
        The set of tokens to exclude from mutations.
    alpha : float, optional
        Alpha parameter for soft reconstruction, default 1.0.
    skip_first_position : bool, optional
        Skip mutations at the first position, default True.

    Returns
    -------
    list[MutatedSequence]
        The proposed mutant sequences.
    """
    # Convert logits to probabilities
    probs = scipy.special.softmax(model_output.logits, axis=1)

    # Get vocab mappings
    token_to_id = model.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in token_to_id.items()}

    # Collect mutant sequences
    mutants = []

    for i, wildtype in enumerate(sequence.sequence):
        # Skip first position mutations if required
        if skip_first_position and (i == 0):
            continue

        # Get token index for the wildtype amino acid
        wildtype_index = token_to_id.get(wildtype, None)
        if wildtype_index is None:
            continue  # Skip unknown characters

        # Extract probabilities for the residue
        residue_probs = probs[i]

        # Get mutant prediction
        mutant_index = np.argmax(residue_probs)
        mutant_token = id_to_token.get(mutant_index, None)

        # Ensure valid mutation
        if mutant_token is None or mutant_token == wildtype or mutant_token in exclude:
            continue

        # Get probabilities of wildtype and mutant
        mutant_prob = residue_probs[mutant_index]
        wildtype_prob = residue_probs[wildtype_index]

        # Accept mutation if mutant probability is significantly higher
        if mutant_prob > alpha * wildtype_prob:
            mutated_sequence = MutatedSequence(
                sequence=sequence,
                mutation=Mutation(i, wildtype, mutant_token),
            )
            mutants.append(mutated_sequence)

    return mutants
'''

def mutate_sequence(
    sequence: Sequence,
    model: LoRAEsm2ProteinLanguageModel,
    model_output: ProteinLanguageModelOutput,
    exclude: set[str],
    alpha: float = 0.8,  # Increase alpha for higher confidence
    skip_first_position: bool = True,
) -> list[MutatedSequence]:
    """Mutate a sequence with improved biological filtering."""
    # Convert logits to probabilities
    probs = scipy.special.softmax(model_output.logits, axis=1)
    
    # Get vocab mappings
    token_to_id = model.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    # Standard amino acids only
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    # Exclude all non-standard amino acids
    exclude = exclude.union(set([t for t in token_to_id.keys() 
                              if len(t) == 1 and t not in valid_aa]))
    # Also exclude special tokens
    exclude.add("<eos>")
    exclude.add("<pad>")
    
    # Collect mutant sequences
    mutants = []
    
    for i, wildtype in enumerate(sequence.sequence):
        # Skip first position mutations if required
        if skip_first_position and (i == 0):
            continue
            
        # Get token index for the wildtype amino acid
        wildtype_index = token_to_id.get(wildtype, None)
        if wildtype_index is None or wildtype not in valid_aa:
            continue  # Skip unknown or non-standard amino acids
            
        # Extract probabilities for the residue
        residue_probs = probs[i]
            
        # Instead of just taking argmax, consider top-k candidates
        top_indices = np.argsort(residue_probs)[::-1][:10]  # Top 10
            
        for mutant_index in top_indices:
            mutant_token = id_to_token.get(mutant_index, None)
                
            # More stringent filtering
            if (mutant_token is None or 
                mutant_token == wildtype or 
                mutant_token in exclude or
                mutant_token not in valid_aa):
                continue
                    
            # Get probabilities
            mutant_prob = residue_probs[mutant_index]
            wildtype_prob = residue_probs[wildtype_index]
                
            # Only accept high-confidence mutations
            if mutant_prob > alpha * wildtype_prob:
                mutated_sequence = MutatedSequence(
                    sequence=sequence,
                    mutation=Mutation(i, wildtype, mutant_token),
                )
                mutants.append(mutated_sequence)
                break  # Only take the highest probability valid mutation
    
    return mutants


class EnsemblePLM:
    """Ensemble of protein language models."""

    def __init__(
        self,
        models: list[Esm2ProteinLangaugeModel],
        consensus_threshold: int = 1,
        exclude: set[str] | None = None,
    ):
        """Initialize the ensemble of protein language models.

        Parameters
        ----------
        models : list[Esm2ProteinLangaugeModel]
            The list of protein language models to form the ensemble.
        consensus_threshold : int, optional
            The number of models needed to form consensus for a given
            mutation, by default uses all proposed mutations of each model.
            Increase to require more models to agree on a mutation.
        exclude : set[str], optional
            The set of tokens to exclude from the mutations, by default
            excludes 'BJOUXZ-.'.
        """
        self.models = models
        self.consenus_threshold = consensus_threshold

        # Set the tokens to exclude from the mutations
        self.exclude = set('BJOUXZ-.') if exclude is None else exclude

        # Set the name of the ensemble
        model_names = ', '.join(model.name for model in self.models)
        self.name = f'Ensemble: {model_names}'

    def mutate(
        self, model: Esm2ProteinLangaugeModel, sequences: list[Sequence]
    ) -> list[MutatedSequence]:
        """Mutate the sequences using a language model."""
        # Run the model on the sequences
        model_outputs = model.infer(sequences)

        # Mutate the sequences
        mutated_sequences = []
        for sequence, model_output in zip(sequences, model_outputs):
            mutated_sequences.extend(
                mutate_sequence(sequence, model, model_output, self.exclude)
            )

        return mutated_sequences

    def infer(self, sequences: list[Sequence]) -> list[MutatedSequence]:
        """Compute the consensus mutation using the models.

        Parameters
        ----------
        sequences : list[str]
            The sequences to get the model output for.

        Returns
        -------
        list[MutatedSequence]
            The consensus mutated sequences.
        """
        # Store all proposed mutants for each sequence
        all_mutants: dict[Sequence, list[MutatedSequence]] = defaultdict(list)

        # Compute the mutations for each model
        for model in self.models:
            # Get the proposed mutants for each sequence
            proposed_mutants = self.mutate(model, sequences)

            # Group the mutations by the original sequence
            for mutant in proposed_mutants:
                all_mutants[mutant.sequence].append(mutant)

        # Compute the consensus mutations
        consensus_mutants: list[MutatedSequence] = []

        # Iterate over each sequence
        for sequence, mutants in all_mutants.items():
            # Get the consensus mutations
            consensus_mutations = self._get_consensus_mutations(mutants)

            # Aggregate the model outputs
            plm_outputs = {}
            for mutant in mutants:
                plm_outputs.update(mutant.plm_outputs)

            for mutation in consensus_mutations:
                # Create the consensus mutated sequence
                mutant = MutatedSequence(
                    sequence=sequence,
                    #plm_outputs=plm_outputs,
                    mutation=mutation,
                )

                # Store the consensus mutated sequence
                consensus_mutants.append(mutant)

        return consensus_mutants

    def _get_consensus_mutations(
        self, mutants: list[MutatedSequence]
    ) -> list[Mutation]:
        """Get the consensus mutations from a list of mutated sequences."""
        # Collect each proposed mutation
        mutations = [mutant.mutation for mutant in mutants]

        # Count the number of times each mutation occurs
        mutation_count = Counter(mutations)

        # Get the mutations that occur more than the consensus threshold
        consensus_mutations = [
            mutation
            for mutation, count in mutation_count.items()
            if count >= self.consenus_threshold
        ]

        return consensus_mutations
