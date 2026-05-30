python visSalFormer/generator.py 
--ckpt 'visSalFormer/VisSalFormer_weights.tar'
--img_dir ChartQA_data/test/png 
--json_path ChartQA_data/test/test_human.json
--max_samples 100
--output_dir ./saliency_maps/ChartQA_test

 
