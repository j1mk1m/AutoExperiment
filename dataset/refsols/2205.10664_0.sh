cd code
cd classification
mv model_onp.py model.py
python train.py --dataset ONP --learning_rate 1e-4 --num_rnn_layer 10 --hidden_dim 20