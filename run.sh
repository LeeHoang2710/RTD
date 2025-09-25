python3 -m torch.distributed.run --nproc_per_node 4 --nnodes 1 --node_rank 0 \
--master_addr localhost --master_port 5100 train_text_encoder.py  \
--batch_size 128 \
--output_dir /path/to/your_experiment \
--cirr_dataset_path /path/to/your_dataset/CIRR \
--clip_model_name large \
--validation_steps 100 \
--checkpointing_steps 100 \
--seed 12345 \
--lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
--max_train_steps 2000 \
--phi_checkpoint "/path/to/your_checkpoint/phi_best.pt" \
--caption_dir "/path/to/your_dataset/LLM_triplets.json" \
--learning_rate 1e-5  \
--mixed_precision "fp16" \


python3 validate.py \
--eval-type phi \
--dataset cirr \
--dataset-path /path/to/your_dataset/CIRR \
--phi-checkpoint-name "/path/to/your_checkpoint/phi_best.pt"   \
--text-encoder-checkpoint-name  "/path/to/your_experiment/text_encoder_best.pt"    \
--clip_model_name large

