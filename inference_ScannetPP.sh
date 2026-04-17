# python Geo_inference_noAlign.py \
# --ckpt_path ./ckpt/Geo_noAlign/latest.pth \
# --image_url ./ScannetPP_testlist/ScannetPP_f1e01a_imgs.txt  \
# --test_line_list ./ScannetPP_testlist/ScannetPP_f1e01a_pkls.txt  \
# --mask_url ./data_list/irregular_mask_30_40.txt  \
# --save_url ./tsr_output/S_f1e01a \
# --iterations 5

# python Animatediff_inpainting.py \
#   --img_list ./ScannetPP_testlist/ScannetPP_f1e01a_imgs.txt \
#   --mask_list data_list/irregular_mask_30_40.txt \
#   --line_list tsr_output/S_f1e01a/line \
#   --output_dir results/S_f1e01a \
#   --gpu_id 0

# python eval_video_inpainting_metrics.py \
# --pre_path ./results/S_f1e01a \
# --gt_path  ../ScannetPP/S_test/S_imgs/f1e01af60a/dslr/resized_undistorted_images/ \
# --mask_path ../masks/irregular_mask/mask_rates_30_40 \
# --pix 256 --resize --out ./eval_out/S_f1e01a


python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f34d53_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f34d53_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f34d53 \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f34d53_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f34d53/line \
  --output_dir results/S_f34d53 \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f34d53 \
--gt_path  ../ScannetPP/S_test/S_imgs/f34d532901/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f34d53




python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f36e3e_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f36e3e_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f36e3e \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f36e3e_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f36e3e/line \
  --output_dir results/S_f36e3e \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f36e3e \
--gt_path  ../ScannetPP/S_test/S_imgs/f36e3e1e53/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f36e3e




python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f38b01_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f38b01_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f38b01 \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f38b01_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f38b01/line \
  --output_dir results/S_f38b01 \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f38b01 \
--gt_path  ../ScannetPP/S_test/S_imgs/f38b0108a1/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f38b01





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f5726e_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f5726e_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f5726e \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f5726e_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f5726e/line \
  --output_dir results/S_f5726e \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f5726e \
--gt_path  ../ScannetPP/S_test/S_imgs/f5726eeeb2/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f5726e





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f84708_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f84708_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f84708 \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f84708_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f84708/line \
  --output_dir results/S_f84708 \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f84708 \
--gt_path  ../ScannetPP/S_test/S_imgs/f847086d15/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f84708





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_f9397a_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_f9397a_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_f9397a \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_f9397a_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_f9397a/line \
  --output_dir results/S_f9397a \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_f9397a \
--gt_path  ../ScannetPP/S_test/S_imgs/f9397af4cb/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_f9397a





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_faba6e_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_faba6e_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_faba6e \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_faba6e_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_faba6e/line \
  --output_dir results/S_faba6e \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_faba6e \
--gt_path  ../ScannetPP/S_test/S_imgs/faba6e97d7/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_faba6e





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_fb893f_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_fb893f_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_fb893f \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_fb893f_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_fb893f/line \
  --output_dir results/S_fb893f \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_fb893f \
--gt_path  ../ScannetPP/S_test/S_imgs/fb893ffaf3/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_fb893f





python Geo_inference_noAlign.py \
--ckpt_path ./ckpt/Geo_noAlign/latest.pth \
--image_url ./ScannetPP_testlist/ScannetPP_fe5fe0_imgs.txt  \
--test_line_list ./ScannetPP_testlist/ScannetPP_fe5fe0_pkls.txt  \
--mask_url ./data_list/irregular_mask_30_40.txt  \
--save_url ./tsr_output/S_fe5fe0 \
--iterations 5

python Animatediff_inpainting.py \
  --img_list ./ScannetPP_testlist/ScannetPP_fe5fe0_imgs.txt \
  --mask_list data_list/irregular_mask_30_40.txt \
  --line_list tsr_output/S_fe5fe0/line \
  --output_dir results/S_fe5fe0 \
  --gpu_id 0

python eval_video_inpainting_metrics.py \
--pre_path ./results/S_fe5fe0 \
--gt_path  ../ScannetPP/S_test/S_imgs/fe5fe0a8a4/dslr/resized_undistorted_images/ \
--mask_path ../masks/irregular_mask/mask_rates_30_40 \
--pix 256 --resize --out ./eval_out/S_fe5fe0

