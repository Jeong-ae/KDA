python ./train/featmatch.py \
--config=config/cifar100/[cifar100][test][cnn13][4000].json \
--name=[cifar100][test][cnn13][vit][4000] \
--iters=1 \
--overwrite \
--amp \
--ckpt='teacher_weight/cifar100/cnn-13/best_ckpt'
