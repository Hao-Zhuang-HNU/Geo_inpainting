python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./DL3DV_testlist/DL3DV_fe30a8_imgs.txt  \
--test_line_list ./DL3DV_testlist/DL3DV_fe30a8_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/noAlignFull_fe30a8 \
--iterations 5

python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./DL3DV_testlist/DL3DV_fe30a8_imgs.txt  \
--test_line_list ./DL3DV_testlist/DL3DV_fe30a8_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/noAlignFull_fe30a8_30_40_20it \
--iterations 20

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f5d7c3_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f5d7c3_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f5d7c3 \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f56bbd_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f56bbd_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f56bbd \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f080ed_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f080ed_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f080ed \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f672ea_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f672ea_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f672ea \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f49418_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f49418_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f49418 \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_f81358_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_f81358_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_f81358 \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_fac184_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_fac184_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_fac184 \
# --iterations 5

# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./DL3DV_testlist/DL3DV_fbb7b6_imgs.txt  \
# --test_line_list ./DL3DV_testlist/DL3DV_fbb7b6_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/noAlignFull_fbb7b6 \
# --iterations 5

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_fe30a8_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_fe30a8/line \
#   --output_dir results/noAlignFull_fe30a8 \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f5d7c3_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f5d7c3/line \
#   --output_dir results/noAlignFull_f5d7c3 \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f56bbd_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f56bbd/line \
#   --output_dir results/noAlignFull_f56bbd \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f080ed_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f080ed/line \
#   --output_dir results/noAlignFull_f080ed \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f672ea_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f672ea/line \
#   --output_dir results/noAlignFull_f672ea \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f49418_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f49418/line \
#   --output_dir results/noAlignFull_f49418 \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_f81358_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_f81358/line \
#   --output_dir results/noAlignFull_f81358 \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_fac184_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_fac184/line \
#   --output_dir results/noAlignFull_fac184 \
#   --gpu_id 0

# python Animatediff_inpainting.py \
#   --img_list ./DL3DV_testlist/DL3DV_fbb7b6_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/noAlignFull_fbb7b6/line \
#   --output_dir results/noAlignFull_fbb7b6 \
#   --gpu_id 0