# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply
from torch import Tensor

from models.base import (
    CrossEntropySystem,
    BatchResult,
)
from models.modules import (
    Decoder,
    DecoderLayer,
    Encoder,
    TokenEmbedding,
)
from models.parseq import PARSeq
from nltk import edit_distance
from PIL import Image


class PARSeqRotationAware(PARSeq):

    def __init__(
        self,
        img_size,
        charset,
        max_label_len,
        patch_size,
        embed_dim,
        enc_num_heads,
        enc_mlp_ratio,
        enc_depth,
        dec_num_heads,
        dec_mlp_ratio,
        dec_depth,
        perm_num,
        perm_forward,
        perm_mirrored,
        decode_ar,
        refine_iters,
        dropout,
        device,
    ) -> None:
        super().__init__(
            img_size,
            charset,
            max_label_len,
            patch_size,
            embed_dim,
            enc_num_heads,
            enc_mlp_ratio,
            enc_depth,
            dec_num_heads,
            dec_mlp_ratio,
            dec_depth,
            perm_num,
            perm_forward,
            perm_mirrored,
            decode_ar,
            refine_iters,
            dropout,
            device,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        tgt_query: Optional[Tensor] = None,
        tgt_query_mask: Optional[Tensor] = None,
    ):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder.forward_cross_attn(
            tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask
        )

    def forward_logits_loss(
        self, images: Tensor, labels: List[str]
    ) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self._device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        out = self.forward_loss(images, max_len)
        logits = out[:, :, :-1]
        loss = F.cross_entropy(
            logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id
        )
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel

    def _eval_step(self, batch, validation: bool):
        images, labels = batch

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
            probs = logits.softmax(-1)
            preds, probs = self.tokenizer.decode(probs)
            for pred, prob, gt in zip(preds, probs, labels):
                confidence += prob.prod().item()
                pred = self.charset_adapter(pred)
                # Follow ICDAR 2019 definition of N.E.D.
                ned += edit_distance(pred, gt) / max(len(pred), len(gt))
                if pred == gt:
                    correct += 1
                total += 1
                label_length += len(pred)
            return dict(
                output=BatchResult(
                    total, correct, ned, confidence, label_length, loss, loss_numel
                )
            )
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            out = self.forward_loss(images)
            loss = loss_numel = (
                None  # Only used for validation; not needed at test-time.
            )
            probs = out[:, :, :-1].softmax(-1)
            attn_score = out[:, 0, -1]

            preds, probs = self.tokenizer.decode(probs)
            for pred, prob, gt in zip(preds, probs, labels):
                confidence += prob.prod().item()
                pred = self.charset_adapter(pred)
                # Follow ICDAR 2019 definition of N.E.D.
                if gt is not None:
                    ned += edit_distance(pred, gt) / max(len(pred), len(gt))
                    if pred == gt:
                        correct += 1
                    total += 1
                label_length += len(pred)
            return (
                dict(
                    output=BatchResult(
                        total,
                        correct,
                        ned,
                        confidence,
                        label_length,
                        loss,
                        loss_numel,
                        attn_score,
                    )
                ),
                preds,
            )

    def forward_loss(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        testing = max_length is None
        max_length = (
            self.max_label_length
            if max_length is None
            else min(max_length, self.max_label_length)
        )
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), float("-inf"), device=self._device), 1
        )
        if self.decode_ar:
            tgt_in = torch.full(
                (bs, num_steps), self.pad_id, dtype=torch.long, device=self._device
            )  # 8xnum_step
            tgt_in[:, 0] = self.bos_id
            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out, attn_scores = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )

                # logits = self.head(tgt_out)

                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)

                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze(1).argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full(
                (bs, 1), self.bos_id, dtype=torch.long, device=self._device
            )
            tgt_out, attn_scores = self.decode(tgt_in, memory, tgt_query=pos_queries)

            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[
                torch.triu(
                    torch.ones(
                        num_steps, num_steps, dtype=torch.bool, device=self._device
                    ),
                    2,
                )
            ] = 0
            bos = torch.full(
                (bs, 1), self.bos_id, dtype=torch.long, device=self._device
            )
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(
                    -1
                ) > 0  # mask tokens beyond the first EOS token.
                tgt_out, attn_scores = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    tgt_query=pos_queries,
                    tgt_query_mask=query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        logits = torch.cat([logits, attn_scores], dim=-1)
        return logits

    def forward(self, images, labels):
        # images, labels= batch
        self._device = images.device
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out, attn_scores = self.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask
            )
            logits = self.head(out)
            loss += n * F.cross_entropy(
                logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id
            )
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        # self.log('loss', loss)
        train_acc = self.train_acc(logits, labels)
        train_acc = torch.tensor(train_acc, device=self._device)
        return loss, train_acc

    def forward_export(self, images):
        testing = True
        max_length = self.max_label_length
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)
        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        tgt_mask = query_mask = torch.triu(
            torch.full((num_steps, num_steps), float("-inf"), device=self._device), 1
        )
        if self.decode_ar:
            tgt_in = torch.full(
                (bs, num_steps), self.pad_id, dtype=torch.long, device=self._device
            )  # 8xnum_step
            tgt_in[:, 0] = self.bos_id
            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out, attn_scores = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query=pos_queries[:, i:j],
                    tgt_query_mask=query_mask[i:j, :j],
                )

                # logits = self.head(tgt_out)

                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)

                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze(1).argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full(
                (bs, 1), self.bos_id, dtype=torch.long, device=self._device
            )
            tgt_out, attn_scores = self.decode(tgt_in, memory, tgt_query=pos_queries)

            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[
                torch.triu(
                    torch.ones(
                        num_steps, num_steps, dtype=torch.bool, device=self._device
                    ),
                    2,
                )
            ] = 0
            bos = torch.full(
                (bs, 1), self.bos_id, dtype=torch.long, device=self._device
            )
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == self.eos_id).int().cumsum(
                    -1
                ) > 0  # mask tokens beyond the first EOS token.
                tgt_out, attn_scores = self.decode(
                    tgt_in,
                    memory,
                    tgt_mask,
                    tgt_padding_mask,
                    tgt_query=pos_queries,
                    tgt_query_mask=query_mask[:, : tgt_in.shape[1]],
                )
                logits = self.head(tgt_out)

        logits = logits.softmax(-1)
        logits = torch.cat([logits, attn_scores], dim=-1)
        return logits