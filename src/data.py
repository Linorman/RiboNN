from typing import Dict, Literal, Optional

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

STAGE = Literal["train", "valid", "test", "predict", "attribute"]


class DataFrameDataset(Dataset):
    """Dataset for turning transcript features and TEs into tensors"""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column_pattern: Optional[str],
        max_utr5_len: int,
        max_cds_utr3_len: int,
        max_tx_len: int,
        pad_5_prime: bool,
        split_utr5_cds_utr3_channels: bool,
        label_codons: bool,
        label_3rd_nt_of_codons: bool,
        label_utr5: bool,
        label_utr3: bool,
        label_splice_sites: bool,
        label_up_probs: bool,
        pad_cds: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        if target_column_pattern:
            self.target_column_pattern = target_column_pattern
            if self.target_column_pattern in self.df.columns:
                self.targets = self.df[[self.target_column_pattern]]
            else:
                self.targets = self.df.filter(regex=rf"{self.target_column_pattern}")
        else:
            self.targets = None
        self.pad_5_prime = pad_5_prime
        self.split_utr5_cds_utr3_channels = split_utr5_cds_utr3_channels
        self.label_codons = label_codons
        self.label_3rd_nt_of_codons = label_3rd_nt_of_codons
        self.label_utr5 = label_utr5
        self.label_utr3 = label_utr3
        self.label_splice_sites = label_splice_sites
        self.label_up_probs = label_up_probs
        self.pad_cds = pad_cds

        # Calculate number of input channels
        self.seq_channels = 4 * 3 if self.split_utr5_cds_utr3_channels else 4
        self.label_channels = (
            self.label_codons
            + self.label_utr5
            + self.label_utr3
            + self.label_splice_sites
            + self.label_up_probs
        )

        # Calculate sizes for aligning sequences at the start codon.
        if pad_5_prime:
            # Pad the 5' UTRs
            self.padded_utr5_len = max_utr5_len
            # Pad the 3' UTRs.
            self.padded_tx_len = max_utr5_len + max_cds_utr3_len
        # Calculate sizes for aligning sequences at the 5' end.
        else:
            # Pad the 3' UTRs.
            self.padded_tx_len = max_tx_len

        self.base_index = {
            "A": 0,
            "T": 1,
            "C": 2,
            "G": 3,
        }

        self._base_lut = np.full(256, -1, dtype=np.int64)
        self._base_lut[ord("A")] = 0
        self._base_lut[ord("T")] = 1
        self._base_lut[ord("C")] = 2
        self._base_lut[ord("G")] = 3

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        # Initiate a tensor to save encodings for sequence, labels, etc.
        x = torch.zeros(
            (self.seq_channels + self.label_channels, self.padded_tx_len),
            dtype=torch.float32,
        )

        original_tx_seq = self.df.tx_sequence[i]
        original_utr5_len = self.df.utr5_size[i]
        original_cds_len = self.df.cds_size[i]
        original_tx_len = self.df.tx_size[i]
        if self.label_splice_sites:
            ss = self.df.splice_sites.values[i]
        if self.label_up_probs:
            probs = self.df.up_prob.values[i].split(";")[:original_tx_len]

        # Encode sequences
        if (
            self.pad_5_prime
        ):  # Padding the 5' end and align sequences at the start codon.
            if self.split_utr5_cds_utr3_channels:
                # Encode 5'UTR
                utr5_seq = original_tx_seq[:original_utr5_len]
                if utr5_seq:
                    base_idx = self._base_lut[np.frombuffer(utr5_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_utr5_len, dtype=torch.int64) + (self.padded_utr5_len - original_utr5_len)
                    channels = torch.from_numpy(base_idx)  # int64
                    x[channels, positions] = 1
                # Encode CDS
                cds_seq = original_tx_seq[original_utr5_len : original_utr5_len + original_cds_len]
                if cds_seq:
                    base_idx = self._base_lut[np.frombuffer(cds_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_cds_len, dtype=torch.int64) + self.padded_utr5_len
                    channels = torch.from_numpy(base_idx) + 4
                    x[channels, positions] = 1
                # Encode 3'UTR
                utr3_seq = original_tx_seq[original_utr5_len + original_cds_len : original_tx_len]
                utr3_len = original_tx_len - original_utr5_len - original_cds_len
                if utr3_seq:
                    base_idx = self._base_lut[np.frombuffer(utr3_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(utr3_len, dtype=torch.int64) + (self.padded_utr5_len + original_cds_len)
                    channels = torch.from_numpy(base_idx) + 8
                    x[channels, positions] = 1
            else:
                # Encode the entire transcript in 4 channels
                if original_tx_seq:
                    base_idx = self._base_lut[np.frombuffer(original_tx_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_tx_len, dtype=torch.int64) + (self.padded_utr5_len - original_utr5_len)
                    channels = torch.from_numpy(base_idx)
                    x[channels, positions] = 1

            # Encode labels
            row_index = self.seq_channels
            if self.label_utr5:
                x[
                    row_index,
                    self.padded_utr5_len - original_utr5_len : self.padded_utr5_len,
                ] = 1
                row_index += 1

            if self.label_codons:
                start = self.padded_utr5_len
                stop = start + original_cds_len - 3
                if stop >= start:
                    indices = torch.arange(start, stop + 1, 3, dtype=torch.int64)
                    x[row_index, indices] = 1
                row_index += 1

            if self.label_utr3:
                x[
                    row_index,
                    (self.padded_utr5_len + original_cds_len) : (
                        self.padded_utr5_len - original_utr5_len + original_tx_len
                    ),
                ] = 1
                row_index += 1

            if self.label_splice_sites:
                if isinstance(
                    ss, str
                ):  # Empty strings are turned into np.nan, which is a float
                    ss_idx = np.fromstring(ss, sep=";")
                    if ss_idx.size:
                        indices = torch.from_numpy(ss_idx.astype(np.int64))
                        indices = indices + (self.padded_utr5_len - original_utr5_len) - 1
                        x[row_index, indices] = 1
                row_index += 1

            # Encode structure
            if self.label_up_probs:
                prob_arr = np.fromstring(self.df.up_prob.values[i], sep=";")[:original_tx_len]
                if prob_arr.size:
                    indices = torch.arange(prob_arr.size, dtype=torch.int64) + (self.padded_utr5_len - original_utr5_len) - 1
                    x[row_index, indices] = torch.from_numpy(prob_arr.astype(np.float32))

        else:  # Not padding the 5' end. Simply align sequences at the 5' end.
            if self.split_utr5_cds_utr3_channels:
                # Encode 5'UTR
                utr5_seq = original_tx_seq[:original_utr5_len]
                if utr5_seq:
                    base_idx = self._base_lut[np.frombuffer(utr5_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_utr5_len, dtype=torch.int64)
                    channels = torch.from_numpy(base_idx)
                    x[channels, positions] = 1
                # Encode CDS
                cds_seq = original_tx_seq[original_utr5_len : original_utr5_len + original_cds_len]
                if cds_seq:
                    base_idx = self._base_lut[np.frombuffer(cds_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_cds_len, dtype=torch.int64) + original_utr5_len
                    channels = torch.from_numpy(base_idx) + 4
                    x[channels, positions] = 1
                # Encode 3'UTR
                utr3_seq = original_tx_seq[original_utr5_len + original_cds_len : original_tx_len]
                utr3_len = original_tx_len - original_utr5_len - original_cds_len
                if utr3_seq:
                    base_idx = self._base_lut[np.frombuffer(utr3_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(utr3_len, dtype=torch.int64) + (original_utr5_len + original_cds_len)
                    channels = torch.from_numpy(base_idx) + 8
                    x[channels, positions] = 1
            else:
                # Encode the entire transcript in 4 channels
                if original_tx_seq:
                    base_idx = self._base_lut[np.frombuffer(original_tx_seq.encode("ascii"), dtype=np.uint8)]
                    positions = torch.arange(original_tx_len, dtype=torch.int64)
                    channels = torch.from_numpy(base_idx)
                    x[channels, positions] = 1

            # Encode labels
            row_index = self.seq_channels
            if self.label_utr5:
                x[row_index, :original_utr5_len] = 1
                row_index += 1

            if self.label_codons:
                start = original_utr5_len
                stop = start + original_cds_len - 3
                if self.label_3rd_nt_of_codons:
                    if stop + 2 >= start + 2:
                        indices = torch.arange(start + 2, stop + 3, 3, dtype=torch.int64)
                        x[row_index, indices] = 1
                else:
                    if stop >= start:
                        indices = torch.arange(start, stop + 1, 3, dtype=torch.int64)
                        x[row_index, indices] = 1
                row_index += 1

            if self.label_utr3:
                x[
                    row_index,
                    (original_utr5_len + original_cds_len) : original_tx_len,
                ] = 1
                row_index += 1

            if self.label_splice_sites:
                if isinstance(
                    ss, str
                ):  # Empty strings are turned into np.nan, which is a float
                    ss_idx = np.fromstring(ss, sep=";")
                    if ss_idx.size:
                        indices = torch.from_numpy(ss_idx.astype(np.int64)) - 1
                        x[row_index, indices] = 1
                row_index += 1

            # Encode structure
            if self.label_up_probs:
                prob_arr = np.fromstring(self.df.up_prob.values[i], sep=";")[:original_tx_len]
                if prob_arr.size:
                    indices = torch.arange(prob_arr.size, dtype=torch.int64) - 1
                    x[row_index, indices] = torch.from_numpy(prob_arr.astype(np.float32))

        if self.targets is None:
            return x

        # Encode targets
        y = self.targets.loc[i, :].values
        y = torch.from_numpy(y).float()

        return x, y


class RiboNNDataModule(pl.LightningDataModule):
    """DataModule for the RiboNN model"""

    def __init__(self, config: Dict):
        super().__init__()
        for k, v in config.items():
            setattr(self, k, v)

        self.df = pd.read_csv(self.tx_info_path, delimiter="\t")
        if "utr5_sequence" in self.df.columns and "cds_sequence" in self.df.columns and "utr3_sequence" in self.df.columns:
            self.df.utr5_sequence = (
                self.df.utr5_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df.cds_sequence = (
                self.df.cds_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df.utr3_sequence = (
                self.df.utr3_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df['tx_sequence'] = self.df.utr5_sequence + self.df.cds_sequence + self.df.utr3_sequence
            self.df["tx_size"] = self.df.tx_sequence.str.len()
            self.df["utr5_size"] = self.df.utr5_sequence.str.len()
            self.df["cds_size"] = self.df.cds_sequence.str.len()
            self.df["utr3_size"] = self.df.utr3_sequence.str.len()
        else:
            assert "tx_sequence" in self.df.columns
            assert "utr5_size" in self.df.columns
            assert "cds_size" in self.df.columns
            self.df.tx_sequence = (
                self.df.tx_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df["utr5_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[:row.utr5_size],
                axis = 1
            )
            self.df["cds_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[row.utr5_size : row.utr5_size + row.cds_size],
                axis = 1
            )
            self.df["utr3_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[row.utr5_size + row.cds_size : ],
                axis = 1
            )
            self.df["tx_size"] = self.df.tx_sequence.str.len()
            self.df["utr3_size"] = self.df.tx_size - self.df.utr5_size - self.df.cds_size

        # Validate cds_sequence
        assert self.df.cds_size.apply(lambda x: x % 3 == 0).all(), "Not all CDS sequences have a length size of 3N!"
        assert self.df.cds_sequence.apply(lambda x: x[-3:] in ("TAA", "TGA", "TAG")).all(), "Not all CDS sequences end with a stop codon!"

        if "tx_id" not in self.df.columns:
            self.df["tx_id"] = range(len(self.df))

        if self.df.tx_size.max() > self.max_seq_len:
            print(
                f"Transcripts longer than {self.max_seq_len} in the input data are removed!"
            )
            self.df = self.df.query("tx_size <= @self.max_seq_len").reset_index(drop=True)

        if self.remove_extreme_txs:
            utr5_len_max = self.df.utr5_size.quantile(0.99)
            cds_len_max = self.df.cds_size.quantile(0.99)
            utr3_len_max = self.df.utr3_size.quantile(0.99)
            print(
                "Top 1% of transcripts with extreme 5'UTR, CDS, or 3'UTR sizes in the input data are removed!",
                "To keep them, set remove_extreme_txs in config/conf.yml to False.",
            )
            self.df = self.df.query(
                "utr5_size <= @utr5_len_max and cds_size <= @cds_len_max and utr3_size <= @utr3_len_max"
            ).reset_index(drop=True)

        if config.get("max_utr5_len", np.Inf) < self.df["utr5_size"].max():
            print(
                f"Transcripts with 5'UTRs longer than {config['max_utr5_len']} in the input data are removed!"
            )
            self.df = self.df.query("utr5_size <= @config['max_utr5_len']").reset_index(
                drop=True
            )
        self.max_utr5_len = config["max_utr5_len"]
        config["max_utr5_len"] = self.max_utr5_len

        self.df["cds_utr3_len"] = self.df.apply(
            lambda row: row.tx_size - row.utr5_size, 1
        )
        if config.get("max_cds_utr3_len", np.Inf) < self.df["cds_utr3_len"].max():
            print(
                f"Transcripts with combined CDS and 3'UTR sizes more than {config['max_cds_utr3_len']} in the input data are removed!"
            )
            self.df = self.df.query(
                "cds_utr3_len <= @config['max_cds_utr3_len']"
            ).reset_index(drop=True)
        self.max_cds_utr3_len = config["max_cds_utr3_len"]
        config["max_cds_utr3_len"] = self.max_cds_utr3_len

        self.max_tx_len = self.df.tx_size.max()

        if self.target_column_pattern:
            targets = self.df.filter(regex=rf"{self.target_column_pattern}")
            self.num_targets = targets.shape[1]
            self.target_names = targets.columns.tolist()
            self.df = self.df.loc[~targets.isnull().all(1), :]

    def get_sequence_length_after_ConvBlocks(self):
        """Calculate the sequence length after the ConvBlocks"""
        # Input sequence length
        if self.pad_5_prime:
            seq_len = self.max_utr5_len + self.max_cds_utr3_len
        else:
            seq_len = self.max_tx_len

        # Sequence length after initial Conv1d
        seq_len = (
            seq_len + 2 * self.conv_padding - self.conv_dilation * (5 - 1) - 1
        ) // self.conv_stride + 1

        for _ in range(self.num_conv_layers):
            # Sequence length after Conv1d
            seq_len = (
                seq_len
                + 2 * self.conv_padding
                - self.conv_dilation * (self.kernel_size - 1)
                - 1
            ) // self.conv_stride + 1
            # Sequence length after MaxPool1d
            seq_len = (seq_len + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1

        return seq_len

    def make_dataloader(
        self,
        stage: STAGE = "train",
        batch_size: int = 32,
        drop_last: bool = False,
    ):
        if stage in ("attribute", "predict"):
            input_df = self.df
            self.target_column_pattern = None
        else:
            input_df = self.df.query("split == @stage")
        dataset = DataFrameDataset(
            input_df,
            self.target_column_pattern,
            self.max_utr5_len,
            self.max_cds_utr3_len,
            self.max_tx_len,
            self.pad_5_prime,
            self.split_utr5_cds_utr3_channels,
            self.label_codons,
            self.label_3rd_nt_of_codons,
            self.label_utr5,
            self.label_utr3,
            self.label_splice_sites,
            self.label_up_probs,
        )

        reorder = stage == "train"

        dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=reorder,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )

        if self.num_workers and self.num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 2

        return DataLoader(
            dataset,
            **dl_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Training stage data loader"""
        return self.make_dataloader("train", self.train_batch_size, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Validation stage data loader"""
        return self.make_dataloader("valid", self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        """Testing stage data loader"""
        return self.make_dataloader("test", self.test_batch_size)

    def attri_dataloader(self) -> DataLoader:
        """Attribution stage data loader"""
        return self.make_dataloader("attribute", self.test_batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Prediction stage data loader"""
        return self.make_dataloader("predict", self.test_batch_size)
