python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path projects/blip/exp_coco_cap_ft.yaml
				# --cfg-run projects/blip/run.yaml \
	            # --cfg-data projects/blip/dataset_cap.yaml \
 		        # --cfg-model projects/blip/model.yaml
				# --evaluate

# python -m torch.distributed.run --nproc_per_node=2 train.py \
# 				--cfg-run projects/blip/run.yaml \
# 	            --cfg-data projects/blip/dataset_cap.yaml \
#  		        --cfg-model projects/blip/model.yaml