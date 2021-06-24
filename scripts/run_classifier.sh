for i in {0..1}
do
    CUDA_VISIBLE_DEVICES=0  python  run_classifier_2.py \
                            --input_path_train='/data/zijian/Bang/mBERT-torch/dataset/lan0_out_train.csv' \
                            --input_path_valid='/data/zijian/Bang/mBERT-torch/dataset/lan0_out_test.csv' \
                            --input_path_test='/data/zijian/Bang/mBERT-torch/dataset/lan0_out_test.csv' \
                            --hidden_dim=768 \
                            --p_dropout=0.2 \
                            --out_dim=2 \
                            --num_epoch=1 \
                            --num_layer=1 \
                            --batch_size=8 \
                            --max_length=100 \
                            --if_specific \
                            --save_salient='/data/zijian/Bang/mBERT-torch/dataset/' \
                            --model_path='/data/zijian/Bang/mBERT-torch/model/' \
                            --out_attention='/data/zijian/Bang/mBERT-torch/results/' \
                            --k $i
done