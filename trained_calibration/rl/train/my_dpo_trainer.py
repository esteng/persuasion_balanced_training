import random 
import wandb 
import pdb 
import numpy as np

from typing import Optional, List
from torch.utils.data import DataLoader
from trl import DPOTrainer
from transformers.trainer_utils import EvalLoopOutput

from trained_calibration.rl.dataset.postprocess import postprocess_extract, postprocess_extract_chat
from trained_calibration.eval.flipflop_eval import evaluate

class MyDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        # take out the eval model
        eval_model = kwargs.pop("eval_model")
        eval_thresh = kwargs.pop("eval_thresh")
        n_eval_batches = kwargs.pop("n_eval_batches")
        # take out chat args 
        is_chat = kwargs.pop("is_chat") 
        boi_token = kwargs.pop("boi_token")
        eoi_token = kwargs.pop("eoi_token")
        super().__init__(*args, **kwargs)
        self.eval_model = eval_model 
        self.eval_thresh = eval_thresh 
        self.n_eval_batches = n_eval_batches

        self.is_chat = is_chat  
        self.boi_token = boi_token
        self.eoi_token = eoi_token


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        
        # Sample and save to game log if requested (for one batch to save time)
        # if self.generate_during_eval:
        if True:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)

            all_accepts, all_corrects = [], []
            # sample n_eval_batches random batches for eval 
            # we have enough examples that overlap should be minimal 
            done_indices = []
            for i in range(self.n_eval_batches):

                random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
                random_indices = [x for x in random_indices if x not in done_indices]
                done_indices.extend(random_indices)

                # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
                try:
                    random_batch_dataset = dataloader.dataset.select(random_indices)
                    random_batch = self.data_collator(random_batch_dataset)
                except (KeyError, IndexError) as e:
                    # if the indices are empty because we already chose these 
                    # happens w/ small eval set 
                    continue

                random_batch = self._prepare_inputs(random_batch)

                try:
                    policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)
                except RuntimeError:
                    print(f"Eval batch OOM from generate, skipping")
                    continue 
                
                # pdb.set_trace()
                # self.log(
                #     {
                #         "game_log": wandb.Table(
                #             columns=["Prompt", "Policy", "Ref Model"],
                #             rows=[
                #                 [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                #                 for prompt, pol, ref in zip(
                #                     random_batch["prompt"], policy_output_decoded, ref_output_decoded
                #                 )
                #             ],
                #         )
                #     }
                # )
                # self.state.log_history.pop()

                # run the reward model 
                if self.eval_model is not None:
                    try:
                        if self.is_chat:
                            out_final, out_answers, rationales = postprocess_extract_chat(random_batch['prompt'], 
                                                                                          policy_output_decoded, 
                                                                                          self.eval_model.model, 
                                                                                          self.eval_model.tokenizer, 
                                                                                          "trivia_qa",
                                                                                          boi_token=self.boi_token,
                                                                                          eoi_token=self.eoi_token)
                        else:
                            out_final, out_answers, rationales = postprocess_extract(random_batch['prompt'], 
                                                                                     policy_output_decoded, 
                                                                                     self.eval_model.model, 
                                                                                     self.eval_model.tokenizer, 
                                                                                     "trivia_qa")

                        scores, corrects, probs = self.eval_model.forward(random_batch['prompt'], policy_output_decoded, out_answers, random_batch['correct_answers'])
                    except RuntimeError:
                        print(f"Eval batch OOM from reward, skipping")
                        continue

                    corrects = np.array(corrects)
                    probs = np.array([x.item() for x in probs]) 
                    accepts = probs > self.eval_thresh
                    all_accepts.extend(accepts)
                    all_corrects.extend(corrects)
        else:
            all_accepts = []
            all_corrects = []

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        if len(all_accepts) == 0:
            final_prec = -1
            final_rec = -1
            final_f1 = -1
        else:

            all_accepts = np.array(all_accepts)
            all_corrects = np.array(all_corrects)
            tp = np.sum(all_accepts * all_corrects)
            fp = np.sum(all_accepts * (1 - all_corrects))
            fn = np.sum((1 - all_accepts) * all_corrects)
            final_prec = tp / (tp + fp)
            final_rec = tp / (tp + fn)
            final_f1 = 2 * final_prec * final_rec / (final_prec + final_rec)

        # add precision, recall, f1
        initial_output.metrics.update({"eval_precision": final_prec, "eval_recall": final_rec, "eval_f1": final_f1})

        return initial_output


    # def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        # """Generate samples from the model and reference model for the given batch of inputs."""

        # # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        # generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        # with generate_context_manager():
        #     policy_output = model.generate(
        #         input_ids=batch["prompt_input_ids"],
        #         attention_mask=batch["prompt_attention_mask"],
        #         max_length=self.max_length,
        #         do_sample=True,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #     )

        #     # if reference_output in batch use that otherwise use the reference model
        #     if "reference_output" in batch:
        #         reference_output = batch["reference_output"]
        #     else:
        #         if self.ref_model is None:
        #             with self.null_ref_context():
        #                 reference_output = self.model.generate(
        #                     input_ids=batch["prompt_input_ids"],
        #                     attention_mask=batch["prompt_attention_mask"],
        #                     max_length=self.max_length,
        #                     do_sample=True,
        #                     pad_token_id=self.tokenizer.pad_token_id,
        #                 )
        #         else:
        #             reference_output = self.ref_model.generate(
        #                 input_ids=batch["prompt_input_ids"],
        #                 attention_mask=batch["prompt_attention_mask"],
        #                 max_length=self.max_length,
        #                 do_sample=True,
        #                 pad_token_id=self.tokenizer.pad_token_id,
        #             )

        # policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        # policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        # reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        # reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        # return policy_output_decoded, reference_output_decoded


class FlipFlopDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        extractor_model = kwargs.pop("extractor_model")
        extractor_tokenizer = kwargs.pop("extractor_tokenizer")
        super().__init__(*args, **kwargs)
        self.extractor_model = extractor_model
        self.extractor_tokenizer = extractor_tokenizer

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """


        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        data, acc = evaluate(self.tokenizer, self.model, self.extractor_tokenizer, self.extractor_model, dataloader, self.args.eval_batch_size, use_dataloader=True)

        # pdb.set_trace()
        initial_output.metrics.update({"eval_flipflop_acc": acc})

        return initial_output